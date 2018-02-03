import json
import Operations.Operations as ops

class DataSetLoader:

    def __init__(self, file):
        self.dataset_file = file
        self.raw_reviews = []


    def get_all_reviews(self):
        '''
        loads the data-set given in a
        :param data_set:
        '''
        data_frame = ops.get_data_frame(self.dataset_file)
        return data_frame.reviewText







