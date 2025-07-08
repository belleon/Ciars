from flask import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)




def linear_regression(data, features, target):
    # преобразование данных
    X = data[features].values
    y = data[target].values
    # разделение на трнировочные и тестовые данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # создание объекта модели линейной регрессии
    model = LinearRegression()

    # обучение модели на тренировочных данных
    model.fit(X_train, y_train)

    # получение предсказаний для тестовых данных
    y_pred = model.predict(X_test)
    y_test = np.where(y_test == 0, 1e-6, y_test)

    # оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real Values', 'y': 'Predictions'})
    return fig,mae,mse


def logistic_regression(data, features, target):
    # преобразование данных
    eu_sales_mean = data['EU_Sales'].mean()
    data['eu_sales_bin'] = np.where(data['EU_Sales'] > eu_sales_mean, 1, 0)
    X = data[features].values
    y = data['eu_sales_bin'].values

    # разделение на тренировочные и тестовые данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # создание объекта модели логистической регрессии
    model = LogisticRegression()

    # обучение модели на тренировочных данных
    model.fit(X_train, y_train)

    # получение предсказаний для тестовых данных
    y_pred = model.predict(X_test)

    # оценка качества модели
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    graph = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real Values', 'y': 'Predictions'})
    return graph,accuracy,precision,recall,f1

def k_means(data, columns, n_clusters=2, n_init=10, max_iter=300):
    X = data[columns].values
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    kmeans.fit(X)
    labels = kmeans.labels_


    return labels
def anomaly_detection(data, columns, kernel='rbf', nu=0.1, gamma='scale'):
    X = data[columns].values
    clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    clf.fit(X)
    y_pred = clf.predict(X)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 500),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 500))

    # Calculate decision function and reshape Z
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create contour plot
    fig = go.Figure()

    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='blues',
                             contours=dict(start=Z.min(), end=0, size=6)))
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='blues',
                             contours=dict(start=0, end=Z.max(), size=2),
                             showlegend=False))

    fig.add_trace(go.Scatter(x=X[y_pred == 1, 0], y=X[y_pred == 1, 1], mode='markers',
                             marker=dict(color='white', size=6, line=dict(color='black', width=0.5)),
                             name='observations'))
    fig.add_trace(go.Scatter(x=X[y_pred == -1, 0], y=X[y_pred == -1, 1], mode='markers',
                             marker=dict(color='black', size=6, line=dict(color='black', width=0.5)),
                             name='outliers'))

    fig.update_layout(xaxis_range=[X[:, 0].min(), X[:, 0].max()],
                      yaxis_range=[X[:, 1].min(), X[:, 1].max()],
                      xaxis_title='Feature 1', yaxis_title='Feature 2',
                      title='One-Class SVM')
    fig.update_traces(showlegend=False)
    return fig


def tune_model_hyperparameters(data, model=Ridge(), cv=5):
    param_grid = {
        'alpha': [0.1, 1, 10],
    }
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv)
    X = np.array(data['NA_Sales']).reshape(-1, 1)
    y = np.array(data['EU_Sales'])
    grid_search.fit(X, y)

    # Plot the learning curve
    train_sizes, train_scores, validation_scores = learning_curve(grid_search.best_estimator_, X, y, cv=cv)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Learning Curve", "Validation Curve"))
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores.mean(axis=1), mode='lines', name='Training score'), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_sizes, y=validation_scores.mean(axis=1), mode='lines', name='Cross-validation score'),
                  row=1, col=1)
    fig.update_xaxes(title_text="Training set size", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)

    # Plot the validation curve
    param_name, param_range = list(param_grid.items())[0]
    train_scores, validation_scores = validation_curve(grid_search.best_estimator_, X, y, param_name=param_name,
                                                       param_range=param_range, cv=cv)
    fig.add_trace(go.Scatter(x=param_range, y=train_scores.mean(axis=1), mode='lines', name='Training score'), row=1, col=2)
    fig.add_trace(go.Scatter(x=param_range, y=validation_scores.mean(axis=1), mode='lines', name='Cross-validation score'),
                  row=1, col=2)
    fig.update_xaxes(title_text=param_name, row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_layout(title='Hyperparameter Tuning', showlegend=True)
    return fig


@app.route("/linr", methods=['GET', 'POST'])
def accuaract():
    '''Функция выводит ошибки'''
    uploaded_file = request.get_json()['file']
    if uploaded_file != '':
        uploaded_file = pd.read_csv(uploaded_file)
        graph,mae,mse=linear_regression(uploaded_file, [request.get_json()["x_column"],request.get_json()["x_column"]],request.get_json()["y_column"])
        return{'mae':mae,
               "mse":mse}


@app.route('/plotlinr', methods=['GET', 'POST'])
def graph():
        '''Функция выводит график линейной регрессии'''
        uploaded_file = request.get_json()['file']
        if uploaded_file != '':
            uploaded_file = pd.read_csv(uploaded_file)
            graph,mae,mse=linear_regression(uploaded_file, [request.get_json()["x_column"],request.get_json()["x_column"]],request.get_json()["y_column"])
            return plotly.io.to_json(graph,pretty=True)


@app.route('/plotLR', methods=['GET', 'POST'])
def graph1():
    '''Функция выводит график логистической регрессии'''
    uploaded_file = request.get_json()['file']
    if uploaded_file != '':
        fig,accuaracy,precisiom,recall,f1= logistic_regression(pd.read_csv(uploaded_file), [request.get_json()["x_column"],request.get_json()["x_column"]],request.get_json()["y_column"])
        graph = plotly.io.to_json(fig, pretty=True)
        return graph


@app.route('/LR',methods=['GET','POST'])
def Log():
    '''Функция выводит ошибки логистической регрессии'''
    uploaded_file = request.get_json()['file']
    if uploaded_file!='':
        uploaded_file=pd.read_csv(uploaded_file)
        fig,accuracy,precision,recall,f1=logistic_regression(uploaded_file, [request.get_json()["x_column"],request.get_json()["x_column"]],request.get_json()["y_column"])
        return {
            'Ac':accuracy,
            "Precision":precision,
            "Recall":recall,
            "F1_score":f1

        }
@app.route('/k_means',methods=['GET','POST'])
def k_graph():
    uploaded_file,x_column,y_column=pd.read_csv(request.get_json()['file']),request.get_json()["x_column"],request.get_json()["y_column"]
    labels = k_means(uploaded_file, [x_column,y_column], n_clusters=2)
    uploaded_file ['cluster'] = labels
    graph=px.scatter(x=uploaded_file[x_column],y=uploaded_file[y_column],color=uploaded_file['cluster'])
    return plotly.io.to_json(graph,pretty=True)

@app.route('/anomaly',methods=['GET','POST'])
def anomaly():
    uploaded_file=pd.read_csv(request.get_json()['file'])
    y_column,x_column=request.get_json()['y_column'], request.get_json()['x_column']
    fig=anomaly_detection(uploaded_file,[x_column,y_column])
    return plotly.io.to_json(fig,pretty=True)

@app.route('/tune_model',methods=['GET','POST'])
def tune():
    uploaded_file=pd.read_csv(request.get_json()['file'])
    graph=tune_model_hyperparameters(uploaded_file)
    return plotly.io.to_json(graph)

if __name__ == '__main__':
    app.run(debug=True)