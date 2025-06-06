import json, asyncio, base64
from client import call_legiscan_api
from bs4 import BeautifulSoup
from collections import defaultdict

def get_bill_text(self, choice):
    masterList = self.holdData[1:]

    billDatas = []
    print("Number of Bills:",len(masterList))

    try:
        with open("../json/bills.json", "r") as file:
            billDatas = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(e)
        for billSummary in masterList[:100]:
            billDatas.append(call_legiscan_api("getBill", id=billSummary["bill_id"])["bill"])
        try:
            with open("../json/bills.json", "w") as file:
                json.dump(billDatas, file, indent=2)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(e)
            billDatas=[]
    except Exception as e:
        print(e)
    
    if choice=="billText":
        billTexts = asyncio.run(self.async_process_all_bill_texts(billDatas))
        self.async_process_all_bill_texts
        try:
            with open("../json/billTexts.json", "w") as file:
                json.dump(billTexts, file, indent=2)
        except Exception as e:
            print(e)

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
