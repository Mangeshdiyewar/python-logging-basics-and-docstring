import pandas as pd
from  utlis.all_utils import prepare_data , save_plot
from  utlis.model import Perceptron

def main(data,modelName,plotName,eta,epochs):
    df_OR = pd.DataFrame(data)
    x, y = prepare_data(df_OR)

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(x, y)

    _ = model_OR.total_loss()

    model_OR.save(filename=modelName, model_dir="model_or")

    save_plot(df_OR, model_OR, filename=plotName)


if __name__ == "__main__":


  OR = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 1, 1, 1]

  }
  ETA =0.3
  EPOCHS =10
  main(data=OR, modelName="or.model",plotName= "or.png", eta =ETA,epochs =EPOCHS)


