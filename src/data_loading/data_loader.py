import pandas as pd

def load_data(path):
    data = pd.read_csv(path, encoding="latin-1", header=None)
    data.columns = ["sentiment", "id", "date", "query", "user", "text"]
    data = data[["sentiment", "text", "date"]]
    data.loc[:,"sentiment"] = data["sentiment"].replace({0: 0, 4: 1})  # 0 = Negativo, 1 = Positivo
    return data