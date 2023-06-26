import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir('e:/Vscode_files/End_to_End_Ml_project_Facemask')
print(os.getcwd())


class data_ingestion_config():
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str  = os.path.join('artifacts',"test.csv")
    raw_data_path:str   = os.path.join('artifacts',"raw.csv")

class data_ingestion():
    def __init__(self):
        self.ingestion_config = data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info('enter the data ingestion component')
        try:
            image_names = os.listdir('downloads/Images')
            labels = []
            for image in image_names:
                name = image.split('.')[0]
                if name == 'with_mask':
                    labels.append('1')
                else:
                    labels.append('0')
            df = pd.DataFrame({'file_name':image_names,'labels':labels})
            logging.info('Data frame created using images directory')
            print(df.shape)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('raw data dataframe saved')

            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42,stratify=df['labels'])
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            train_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("train and test datas are saved...")
            print(train_data.shape)
            print(test_data.shape)
            

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = data_ingestion()
    obj.initiate_data_ingestion()
