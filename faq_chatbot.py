# ============================================================
# PROJECT 5: FAQ Chatbot (NLP-based Retrieval)
# Skills: Python, NLP, TF-IDF, Cosine Similarity
# ============================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# ── FAQ Knowledge Base ──
FAQ = {
    "What is machine learning?":
        "Machine Learning is a branch of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",

    "What is natural language processing?":
        "Natural Language Processing (NLP) is a field of AI that deals with the interaction between computers and humans using natural language. It helps computers understand, interpret, and generate human language.",

    "What is a neural network?":
        "A neural network is a series of algorithms that mimics the operations of a human brain to recognize relationships between vast amounts of data. It consists of layers of nodes (neurons) that process information.",

    "What is deep learning?":
        "Deep Learning is a subset of machine learning that uses neural networks with many layers (deep networks) to learn representations of data. It is especially powerful for image, audio, and text tasks.",

    "What is Python?":
        "Python is a high-level, versatile programming language widely used in data science, machine learning, and web development. It is known for its simple syntax and large ecosystem of libraries like NumPy, Pandas, and Scikit-learn.",

    "What is TF-IDF?":
        "TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic used in NLP to reflect how important a word is to a document in a collection. It is widely used in text classification and search.",

    "What is Naive Bayes?":
        "Naive Bayes is a simple probabilistic machine learning algorithm based on Bayes Theorem. It assumes features are independent of each other, making it fast and efficient for text classification tasks like spam detection.",

    "What is sentiment analysis?":
        "Sentiment analysis is the use of NLP and ML to identify and extract subjective information from text, such as opinions and emotions. It is used to determine whether text expresses positive, negative, or neutral sentiment.",

    "How do I start learning machine learning?":
        "Start with Python basics, then learn NumPy and Pandas for data handling. Study linear algebra and statistics fundamentals. Practice with Scikit-learn for classical ML, and explore platforms like Kaggle for real datasets and competitions.",

    "What are common machine learning algorithms?":
        "Common ML algorithms include Linear Regression, Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Naive Bayes, and Gradient Boosting methods like XGBoost.",

    "What is overfitting in machine learning?":
        "Overfitting occurs when a model learns the training data too well, including its noise and outliers, causing poor performance on new unseen data. It can be addressed using techniques like cross-validation, regularization, and dropout.",

    "What is the difference between supervised and unsupervised learning?":
        "Supervised learning uses labelled data where the algorithm learns from input-output pairs. Unsupervised learning works with unlabelled data and finds hidden patterns or groupings on its own, like clustering algorithms.",

    "What is a confusion matrix?":
        "A confusion matrix is a table used to evaluate the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives for each class.",

    "What is logistic regression?":
        "Logistic Regression is a supervised ML algorithm used for binary classification. Despite its name, it outputs probabilities using the sigmoid function to classify inputs into two categories like spam/ham or yes/no.",

    "What is GitHub?":
        "GitHub is a web-based platform for version control and collaboration. It lets developers store, manage, and track changes in their code using Git. It is widely used for open-source projects and portfolio building.",
}

questions = list(FAQ.keys())
answers   = list(FAQ.values())

# ── Build TF-IDF Index ──
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
faq_vectors = vectorizer.fit_transform(questions)

def get_response(user_query, threshold=0.05):
    """Find the best matching FAQ answer using cosine similarity."""
    query_clean = re.sub(r'[^a-z0-9\s]', '', user_query.lower()).strip()
    if not query_clean:
        return "I didn't understand that. Could you rephrase your question?"

    query_vec = vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vec, faq_vectors).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return ("I'm not sure I have an answer for that.\n"
                "Try asking about: machine learning, NLP, Python, neural networks, "
                "deep learning, GitHub, or common algorithms.")

    return answers[best_idx]

# ── CLI Chatbot Loop ──
print("=" * 55)
print("         PROJECT 5: FAQ CHATBOT")
print("=" * 55)
print("🤖 Hi! I'm an AI-powered FAQ Chatbot.")
print("   Ask me anything about ML, NLP, Python, and AI!")
print("   Type 'quit' or 'exit' to stop.\n")
print("📚 Topics I know: Machine Learning, NLP, Python,")
print("   Deep Learning, Algorithms, GitHub, and more.\n")
print("-" * 55)

# ── Demo mode: show sample Q&A ──
demo_questions = [
    "What is machine learning?",
    "How can I learn ML?",
    "What is TF-IDF?",
    "Tell me about GitHub",
    "What is overfitting?",
]

print("\n🧪 Demo Mode — Sample Questions & Answers:\n")
for q in demo_questions:
    answer = get_response(q)
    print(f"👤 You : {q}")
    print(f"🤖 Bot : {answer[:120]}{'...' if len(answer)>120 else ''}")
    print()

print("-" * 55)
print("\n💬 Interactive Mode (type your question below):")
print("   [In a terminal, this will accept live input]")
print("\n   Example: python faq_chatbot.py")
print("   Then type your question and press Enter.\n")

# ── Interactive mode (works in terminal) ──
import sys
if sys.stdin.isatty():
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("🤖 Bot: Goodbye! Happy learning! 👋")
                break
            if user_input:
                response = get_response(user_input)
                print(f"🤖 Bot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n🤖 Bot: Goodbye! Happy learning! 👋")
            break
