import joblib
import pandas as pd

def main():

    #  تحميل النموذج المدرب
    model = joblib.load("sentiment_model.pkl")

    #  تحميل TF-IDF vectorizer
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    #  قراءة بيانات جديدة من CSV
    data = pd.read_csv("Customer_Sentiment.csv")

    #  اختيار عمود النص
    texts = data["review_text"]

    #  تحويل النص إلى أرقام باستخدام نفس الـ TF-IDF
    X = vectorizer.transform(texts)

    #  التنبؤ بالمشاعر
    predictions = model.predict(X)

    # إضافة النتائج للبيانات
    data["Predicted_Sentiment"] = predictions

    #  عرض النتائج
    print("Prediction Results:")
    print(data)

    # 9) حفظ النتائج في ملف جديد
    data.to_csv("output_with_predictions.csv", index=False)
    print("\nResults saved to output_with_predictions.csv")


if __name__ == "__main__":
    main()
    