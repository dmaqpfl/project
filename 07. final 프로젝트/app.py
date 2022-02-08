from flask import Flask, render_template, redirect, request, url_for
import test_func
import testpython
import category as ca





app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summary():
    return render_template('summarize.html')

@app.route('/summarize_form', methods=['POST'])
def printText():
    # 원문
    temp = request.form['summarize_textarea']
    print(temp)

    # 카테고리 분류
    text_preprocession= ca.text_preprocession(temp)
    category_model = ca.text_pre(text_preprocession)
    categorizer = ca.categorizer(category_model)
    print(categorizer)

    # 요약 
    result = testpython.summarize_last(temp)
    print(result)

    if result == False:
        result="문장이 너무 짧습니다."
    else :
        result = testpython.summarize_last(temp)
    
    return render_template('summarize.html', summarize_textarea=temp,text_summarize=result,categorizer=categorizer)





if __name__=="__main__":
    app.run(host="127.0.0.1",port="8080",debug=True)