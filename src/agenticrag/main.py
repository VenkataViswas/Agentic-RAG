from graphs.graph import app

def main():
    print("=="*20)
    print(app.invoke(input={"question": "what is agent memory?"}))