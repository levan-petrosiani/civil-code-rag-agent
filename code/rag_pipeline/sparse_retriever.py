from rank_bm25 import BM25Okapi
import re

class SparseRetriever:
    """
    ეს კლასი გამოიყენება ტექსტის ძიებისა და რანჟირებისთვის BM25 ალგორითმის გამოყენებით, რომელიც განკუთვნილია ქართული ტექსტის დამუშავებისთვის.
    __init__(self, chunks):
    ინიციალიზაციის ფუნქცია, რომელიც იღებს ტექსტის ფრაგმენტებს (chunks), ინახავს მათ, ახდენს მათ ტოკენიზაციას (ტექსტის სიტყვებად დაყოფას) და ქმნის BM25Okapi ობიექტს, რომელიც გამოიყენება ძიებისთვის.
    tokenize(self, text):
    მარტივი ტოკენიზატორი, რომელიც ქართულ ტექსტს ასუფთავებს ზედმეტი სფეისებისგან და ყოფს სიტყვებად, აბრუნებს სიტყვების სიას.
    search(self, query, top_k=10):
    ძიების ფუნქცია, რომელიც იღებს მომხმარებლის შეკითხვას (query), ახდენს მის ტოკენიზაციას, აფასებს დოკუმენტებს BM25 ალგორითმის მიხედვით და აბრუნებს top_k რაოდენობის ყველაზე შესაბამისი დოკუმენტის ტექსტს, რანჟირებული ქულების მიხედვით.
    """
    
    def __init__(self, chunks):
        # Preprocess chunks into token lists
        self.documents = [chunk['text'] for chunk in chunks]
        self.tokenized_docs = [self.tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def tokenize(self, text):
        # Simple Georgian tokenizer 
        text = re.sub(r'\s+', ' ', text)
        return text.split(" ")

    def search(self, query, top_k=10):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices]
    

