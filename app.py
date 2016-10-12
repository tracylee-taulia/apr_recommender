from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template
import random
import requests
import pickle
import os
from werkzeug.utils import secure_filename
import pandas as pd
from participation_rate_model import *
from APR_model import *

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/tracy.lee/Documents/Data_Services/web_app/uploads'
EXPORT_FOLDER = '/Users/tracy.lee/Documents/Data_Services/web_app/exports'
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'tsv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPORT_FOLDER'] = EXPORT_FOLDER
# Initialize your app and load your pickled models.
#================================================
# init flask app



# Homepage with form on it.
#================================================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        data = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if data.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if data and allowed_file(data.filename):
            filename = secure_filename(data.filename)
            data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict',
                                    filename=filename))
    return render_template('index.html')
    # return '''
    #
    #
    #     <!DOCTYPE html>
    #         <html>
    #             <head>
    #                 <meta charset="utf-8">
    #                 <title>Data upload</title>
    #             </head>
    #           <body>
    #             <!-- page content -->
    #             <h1>APR Recommender</h1>
    #             <p>
    #                 Please submit your supplier file here.
    #                 <br> <b>The supplier file must come in the standard format
    #                 with standard column headings. </b>
    #             </p>
    #           </body>
    #         </html>
    # <form action="" method=post enctype=multipart/form-data>
    #   <p><input type=file name=file>
    #      <input type=submit value=Upload>
    # </form>
    # '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)






# Once submit is hit, pass info into model, return results.
#================================================
@app.route('/predict<filename>', methods=['GET','POST'])
def predict(filename):
    # get data from request form
    df = pd.read_csv('{}/{}'.format(app.config['UPLOAD_FOLDER'],filename))

    model = pickle.load(open('data/participation_model.pkl', 'r'))
    data = data_transformation(df.copy())
    apr_pr = optimize_apr_pr(data, model)
    df[['Recommended_APR', 'Participation_Rate']] = pd.DataFrame(apr_pr)

    export_filename = 'results_{}'.format(filename)
    export_file = '{}/results_{}'.format(app.config['EXPORT_FOLDER'],filename)
    df.to_csv(export_file)

    return render_template('view.html',filename=export_filename, tables=[df.to_html(index=False)],
    titles = ['na', 'Results'])


@app.route('/export/<filename>', methods=['GET','POST'])
def export(filename):
    return send_from_directory(app.config['EXPORT_FOLDER'],
                               filename)



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=8080, debug=True)
