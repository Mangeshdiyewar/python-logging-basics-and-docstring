import os
import pandas as pd
from  utlis.all_utils import prepare_data , save_plot
from  utlis.model import Perceptron
import logging

gate = "OR gate"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir,"running_logs.log"),
    level= logging.INFO,
    format='[%(asctime)s:%(levelname)s:%(module)s]:%(message)s',
    filemode='a'
    )
def main(data,modelName,plotName,eta,epochs):
    df = pd.DataFrame(data)
    logging.info('this is raw data set:\n{df}')
    x, y = prepare_data(df)

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(x, y)

    _ = model_OR.total_loss()

    model_OR.save(filename=modelName, model_dir="model_or")

    save_plot(df, model_OR, filename=plotName)


if __name__ == "__main__":


  OR = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 1, 1, 1]

  }
  ETA =0.3
  EPOCHS =10
  try:
      logging.info(">>>>> starting training {gate} >>>>")
      main(data=OR, modelName="or.model",plotName= "or.png", eta =ETA,epochs =EPOCHS)
      logging.info(f"<<<<<done training for {gate}<<<<<\n\n")
  except Exception as e:
      logging.exception(e)
      raise e



