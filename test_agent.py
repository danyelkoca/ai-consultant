import os
from agent import ConsultantAgent


def test_analysis():
    # Initialize agent with data path
    data_path = os.path.join(os.path.dirname(__file__), "data", "sales.csv")
    agent = ConsultantAgent(data_path=data_path)

    # Test with a question focused on finding root cause
    question = "What caused the sales decline between April 2024 and April 2025?"
    print(f"\nTesting question: {question}")

    # Get and print answer
    answer = agent.analyze(question)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    test_analysis()
