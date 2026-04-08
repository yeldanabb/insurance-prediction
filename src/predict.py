import joblib
import pandas as pd
import argparse
import os

def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, '..', 'models', 'model.pkl')
    return joblib.load(model_path)

def main():
    parser = argparse.ArgumentParser(description="Предсказание страховых расходов клиента")
    parser.add_argument('--age', type=int, required=True, help='Возраст клиента')
    parser.add_argument('--sex', choices=['male', 'female'], required=True, help='Пол')
    parser.add_argument('--bmi', type=float, required=True, help='Индекс массы тела')
    parser.add_argument('--children', type=int, required=True, help='Количество детей')
    parser.add_argument('--smoker', choices=['yes', 'no'], required=True, help='Курильщик (yes/no)')
    parser.add_argument('--region', choices=['northeast', 'northwest', 'southeast', 'southwest'], required=True, help='Регион')
    args = parser.parse_args()

    model = load_model()
    input_data = pd.DataFrame([{
        'age': args.age,
        'sex': args.sex,
        'bmi': args.bmi,
        'children': args.children,
        'smoker': args.smoker,
        'region': args.region
    }])

    prediction = model.predict(input_data)[0]
    print(f"Ожидаемые расходы: ${prediction:,.2f}")

if __name__ == "__main__":
    main()