from DataLoad.DataLoader import DataSetLoader
from Features.FeatureExtractor import BagofWordFeatureExtractor
import time

def run_process(dataset):
    #get the raw reviews from dataset
    data_set_loader = DataSetLoader(dataset)
    reviews = data_set_loader.get_all_reviews()

    bof_extractor = BagofWordFeatureExtractor()
    corpus = bof_extractor.get_corpus(reviews)
    print("\n\n")
    print(len(corpus))


start_time = time.time()
run_process("/Users/lrravi/Documents/Personal/FBHackathon/data/reviews_Books_5")
end_time = time.time()
print("time taken", (end_time-start_time))
