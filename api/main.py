import json, argparse, pdfplumber
import bill.run as run

from client import call_legiscan_api
from transformers import AutoTokenizer
from datetime import datetime
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
            run.get_bill_text(self,args.command)
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
            with open("../json/master.json", "r") as file:
                self.holdData = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def get_master_list_raw(self):
        data = call_legiscan_api("getMasterListRaw", state="US")
        bills = list(data['masterlist'].values())
        try:
            with open("json/master.json", "w") as file:
                json.dump(bills, file, indent=2)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(e)

    def get_master_list(self):
        print("here")
        data = call_legiscan_api("getMasterList", state="US")
        bills = list(data['masterlist'].values())
        try:
            with open("json/master.json", "w") as file:
                json.dump(bills, file, indent=2)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(e)

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
        except (FileNotFoundError, json.JSONDecodeError) as e:
            billDatas=[]
            print(e)

        print(len(billDatas))
        for bill in billDatas:
            print(f"{bill["doc_id"]} \nNeutral Summary: {bill["final_summary"]} \nLiberal Summary: {bill["liberal_summary"]} \nConservative Summary: {bill["conservative_summary"]}\n\n")

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
