from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'D:/study_data/_testing_image/dogs'

#업로드 HTML 렌더링
@app.route('/')
def render_file():
   return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      #저장할 경로 + 파일명
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      return 'uploads 디렉토리 -> 파일 업로드 성공!'

if __name__ == '__main__':
    #서버 실행
    
   app.run(debug = True)
