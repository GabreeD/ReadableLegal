import base64, json, argparse, pdfplumber, asyncio

from transformers import AutoTokenizer
from datetime import datetime
from client import call_legiscan_api
from bs4 import BeautifulSoup
from collections import defaultdict
from transformers import pipeline

class LegiScanAPI:
    holdData =[]
    def __init__(self):
        # self.get_master_list_raw()
        # self.get_master_list()

        parser = argparse.ArgumentParser(description="Simple To-Do List Manager")
        parser.add_argument("command", choices=["bill", "billText","create", "reset", "resetRAW","analyze","read"])
        parser.add_argument("arg", nargs="?", help="Task description or task number")
        args = parser.parse_args()

        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

        if args.command == "bill" or args.command == "billText":
            self.load_task()
            self.get_bill_text(args.command)
        elif args.command == "reset":
            self.get_master_list()
        elif args.command == "resetRAW":
            self.get_master_list_raw()
        elif args.command == "create":
            self.create_ai_file()
        elif args.command == "analyze":
            self.analyze_data()
        elif args.command == "read":
            self.read_bills()

    def load_task(self):
        try:
            with open("master.json", "r") as file:
                self.holdData = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def get_master_list_raw(self):
        print("here")
        data = call_legiscan_api("getMasterListRaw", state="US")
        bills = list(data['masterlist'].values())

        with open("master.json", "w") as file:
            json.dump(bills, file, indent=2)


    def get_master_list(self):
        print("here")
        data = call_legiscan_api("getMasterList", state="US")
        bills = list(data['masterlist'].values())
        
        with open("master.json", "w") as file:
            json.dump(bills, file, indent=2)

    def get_bill_text(self, choice):
        masterList = self.holdData[1:]

        billDatas = []
        print("Number of Bills:",len(masterList))

        try:
            with open("bills.json", "r") as file:
                billDatas = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            for billSummary in masterList[:100]:
                billDatas.append(call_legiscan_api("getBill", id=billSummary["bill_id"])["bill"])
            with open("bills.json", "w") as file:
                json.dump(billDatas, file, indent=2)
        
        if choice=="billText":
            billTexts = asyncio.run(self.async_process_all_bill_texts(billDatas))

            with open("billTexts.json", "w") as file:
                json.dump(billTexts, file, indent=2)

    async def async_process_all_bill_texts(self, billDatas):
        tasks = []

        for bill in billDatas:
            for textDoc in bill["texts"]:
                tasks.append(self.process_text_doc(bill, textDoc))  

        results = await asyncio.gather(*tasks)
        return results

    async def process_text_doc(self, bill, textDoc):
        billText = await asyncio.to_thread(call_legiscan_api, "getBillText", id=textDoc["doc_id"])
        billText = billText["text"]
        decoded_bytes = base64.b64decode(billText["doc"])

        try:
            decoded_text = decoded_bytes.decode("utf-8")
            soup = BeautifulSoup(decoded_text, "html.parser")
            clean_text = soup.get_text()
            billText["decoded_text"] = clean_text
        except UnicodeDecodeError:
            await asyncio.to_thread(self.extract_pdf_text, decoded_bytes, billText)

        # Group sponsors (cheap, keep as-is)
        grouped_sponsors = defaultdict(list)
        for s in bill["sponsors"]:
            key = f"{s['role']}-{s['party']}"
            val = f"{s['name']} ({key}-{s['district']})"
            grouped_sponsors[key].append(val)

        # Split & summarize in parallel
        chunks = self.split_into_chunks(billText["decoded_text"])
        summaries = await asyncio.gather(*[
            asyncio.to_thread(
                self.summarizer,
                f"Summarize clearly and concisely: {chunk}",
                max_length=150,
                min_length=60,
                do_sample=False
            ) for chunk in chunks
        ])

        partial_summaries = [s[0]["summary_text"] for s in summaries]
        split_text = partial_summaries[0] if len(partial_summaries) == 1 else partial_summaries

        return {
            "doc_id": billText["doc_id"],
            "bill_id": billText["bill_id"],
            "url": billText["url"],
            "decoded_text": {"text": billText["decoded_text"]},
            "analysis_text": split_text,
            "sponsors": dict(grouped_sponsors),
            "progress": bill["progress"],
        }

    def extract_pdf_text(self, decoded_bytes, billText):
        with open("bill.pdf", "wb") as f:
            f.write(decoded_bytes)
        with pdfplumber.open("bill.pdf") as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
        billText["decoded_text"] = text

    def analyze_data(self):

        try:
            with open("./json/billTexts.json", "r") as file:
                billDatas = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Could find billTexts.json")

        for index,bill in enumerate(billDatas):

            base_text=bill["analysis_text"]
            try:
                neutral_prompt = f"Summarize this legislative bill clearly and factually: {base_text}"
                neutral_summary = self.summarizer(neutral_prompt, max_length=150, min_length=100, do_sample=False)[0]["summary_text"]
                billDatas[index]["final_summary"] = neutral_summary

                liberal_prompt = f"Summarize this legislative bill from a liberal perspective, highlighting concerns about individual rights, access to healthcare, and social justice: {base_text}"
                liberal_summary = self.summarizer(liberal_prompt, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
                billDatas[index]["liberal_summary"] = liberal_summary

                conservative_prompt = f"Summarize this legislative bill from a conservative perspective, emphasizing constitutional values, fiscal responsibility, and traditional principles: {base_text}"
                conservative_summary = self.summarizer(conservative_prompt, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
                billDatas[index]["conservative_summary"] = conservative_summary
            except Exception as e:
                billDatas[index]["final_summary"] = f"Summary failed: {str(e)}"

            # Key Points
            # try:
            #     doc = self.nlp(bill["decoded_text"]["text"])
            #     # Score sentences by length and noun density
            #     scored = sorted(
            #         ((sent, len(sent.text), sum(1 for token in sent if token.pos_ == "NOUN")) for sent in doc.sents),
            #         key=lambda tup: (tup[2], tup[1]),  # prioritize noun-rich, longish sentences
            #         reverse=True
            #     )
            #     top_sentences = sorted(scored[:5], key=lambda s: bill["decoded_text"]["text"].find(s[0].text))  # keep original order

            #     keyPoints=[]
            #     for s in top_sentences:
            #         chunk = f"Summarize the following legislative excerpt clearly: {s[0].text.strip()}"
            #         summary_text = self.summarizer(chunk[:1024], max_length=80, min_length=40, do_sample=False)[0]["summary_text"]

            #     #     summary_text = self.summarizer(s[0].text.strip()[:1024], max_length=70, min_length=30, do_sample=False)[0]["summary_text"]
            #     #     print(summary_text)
            #     #     print()
            #         keyPoints.append(summary_text)

            #     # billDatas[index]["key_points"] = [s[0].text.strip() for s in top_sentences]
            #     billDatas[index]["key_points"]= keyPoints
            # except Exception as e:
            #     print(f"Summary failed: {str(e)}")
            
            print(len(base_text))
            print(f"{index+1} out of {len(billDatas)}: {datetime.now()}")

        with open("billTexts.json", "w",encoding="utf-8") as file:
            json.dump(billDatas, file, indent=2,ensure_ascii=False)

    def read_bills(self):
        try:
            with open("../json/billTexts.json", "r") as file:
                billDatas = json.load(file)
                print("Before",len(billDatas))
        except (FileNotFoundError, json.JSONDecodeError):
            billDatas=[]
            print("Couldn't find billTexts.json")

        print(len(billDatas))
        for bill in billDatas:
            print(f"{bill["doc_id"]} \nNeutral Summary: {bill["final_summary"]} \nLiberal Summary: {bill["liberal_summary"]} \nConservative Summary: {bill["conservative_summary"]}\n\n")

    def split_into_chunks(self, text, max_tokens=1024, stride=256):
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            print(start,len(tokens))
            end = min(start + max_tokens, len(tokens))
            chunk = tokens[start:end]
            chunks.append(chunk)
            start += max_tokens - stride  # overlap
        print()
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

class FollowTheMoney:
    # https://www.followthemoney.org/our-data/apis/documentation
    print()

if __name__ == "__main__":
    print()
    print()
    print(" 8====D  ")
    print()
    print()
    LegiScanAPI()
