import pandas as pd
from  utlis.all_utils import prepare_data , save_plot
from  utlis.model import Perceptron

def main(data,modelName,plotName,eta,epochs):
    df_AND = pd.DataFrame(data)
    x, y = prepare_data(df_AND)

    model_AND = Perceptron(eta=eta, epochs=epochs)
    model_AND.fit(x, y)

    _ = model_AND.total_loss()

    model_AND.save(filename=modelName, model_dir="model_AND")

    save_plot(df_AND, model_AND, filename=plotName)


if __name__ == "__main__":


  AND = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 0, 0, 1]

  }
  ETA =0.3
  EPOCHS =10
  main(data=AND, modelName="and_model",plotName= "and.png", eta =ETA,epochs =EPOCHS)


