#########################################################################
# File Name: ann_online_333.py
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Wed 25 Sep 2024 09:28:29 PM CST
#########################################################################
from flask import Flask, render_template, jsonify
from flask import request
import requests

app = Flask(__name__)


@app.route("/")
def index():
    key = request.args.get('key',"")
    print(key)
    return render_template("main.html")

@app.route("/get-links")
def get_links():
    key = request.args.get('key', "")
    print(key)
    #return jsonify(url1="/page1", url2="/page2", url3="/page3")
    # 返回动态生成的链接页面
    return jsonify(url1=f"/page1?key={key}", url2=f"/page2?key={key}", url3=f"/page3?key={key}")

@app.route("/page1")
def page1():
    key = request.args.get('key', '')
    print(key)
    url = "http://9.134.241.141:4000/ann?keyword=" + str(key)
    response = requests.get(url)
    content = response.text
    return content

@app.route("/page2")
def page2():
    key = request.args.get('key', '')
    print(key)
    url = "http://9.134.241.141:5001/ann?keyword=" + str(key)
    response = requests.get(url)
    content = response.text
    return content

@app.route("/page3")
def page3():
    key = request.args.get('key', '')
    print(key)
    url = "http://9.134.241.141:5002/ann?keyword=" + str(key)
    response = requests.get(url)
    content = response.text
    return content

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4002)
        #const response = await fetch("/get-links");
        #const urls = await response.json();
