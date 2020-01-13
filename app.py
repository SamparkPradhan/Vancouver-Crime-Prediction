import json
from flask import Flask, render_template,request
import ast
import folium
from folium.plugins import MarkerCluster
import plotly
import plotly.graph_objs as go
import plotly.express as px
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score, recall_score,f1_score,accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import plotly.graph_objects as go
import lime
import lime.lime_tabular
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
df_crime = pd.read_csv('crime.csv')
df_crime = df_crime[df_crime['YEAR'] >= 2007]
main_filter_yr = '2017'
main_filter_type=[]
main_filter_type.append("Accident")
years_line = []
years_line += df_crime['YEAR'].astype(str).unique().tolist()
years_bar = []
fig="";
model_data="";

#This url render page to main homepage of application.It render page to homePage.html with Crime Data list
@app.route('/')
def mainPage():

    df = pd.read_csv('crime.csv')

    #default selected crimeType
    CrimeTypeList=["Theft"];

    #get list of crimedata based on crime type and year, default CrimeType="Theft" and Year="2017".
    selected_data = getSelectedCrimeData(CrimeTypeList,2017,df)

    #Create map with markers based on selected dataset (CrimeType and Year)
    getMap(selected_data)

    #get LineChart data, Line chart based on selected year and Crime type,default CrimeType="Theft" and Year="2017".
    graphData = getLineChartData("2017", CrimeTypeList)

    graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

    #render page to main HomePage.html with lineData.
    return render_template("homePage.html", data=graphJSON)

#This url is called when user selected different year and Crime type, Based on selected value it will render data to frontend.
@app.route("/getData",methods=['PUT','GET'])
def getDataFunction():
    df = pd.read_csv('crime.csv')
    selectedYear = request.args['selectedYear'];
    selectedCrime=ast.literal_eval(request.args['selectedCrime']);
    df=getSelectedCrimeData(selectedCrime,selectedYear,df);
    getMap(df);
    chart_data = df.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return data

#This url is called to change line graph based on selected crimeType and Year.
@app.route("/getGraphData",methods=['PUT','GET'])
def getGraphData():

    selectedYear = request.args['selectedYear'];
    selectedCrime = ast.literal_eval(request.args['selectedCrime']);
    graphData = getLineChartData(selectedYear, selectedCrime)
    graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# This function create Html which contain the map and markers based on selected year and crimeType.
def getMap(df):

    #Vancouver location Co-ordinates.
    VC_COORDINATES = (49.2827, -123.1207)
    crime_map = folium.Map(location=VC_COORDINATES, zoom_start=8)
    marker = MarkerCluster()

    # This add markers on blank map
    for i in df.itertuples():
        marker.add_child(folium.Marker(location=[i.Latitude, i.Longitude],
                                       popup=i.TYPE))
    crime_map.add_child(marker)
    crime_map.save('static/map.html')
    return crime_map

def getLineChartData(selected_year, multiple_types):
    main_filter_yr = selected_year
    yr_filtered_df = df_crime[df_crime['YEAR'] == int(selected_year)]
    traces = []
    d = dict(
        [(1, 'Jan'), (2, 'Feb'), (3, 'Mar'), (4, 'Apr'), (5, 'May'), (6, 'Jun'), (7, 'Jul'), (8, 'Aug'), (9, 'Sep'),
         (10, 'Oct'), (11, 'Nov'), (12, 'Dec')])

    for selected_type in multiple_types:
        subTypes = getSubfilterType(selected_type)
        subfiltered_df = getSubTypeFilteredDf(subTypes, yr_filtered_df)
        for type in subTypes:
            tracefilter_df =subfiltered_df[subfiltered_df.TYPE == type]
            grouped_df = tracefilter_df.groupby('MONTH').count()
            se = (grouped_df.TYPE)
            final_df = pd.DataFrame({'MONTH': se.index, 'COUNT': se.values})
            monlist = []
            for m in final_df.MONTH:
                monlist.append(d[m])
            final_df['MONTH'] = monlist
            traces.append(go.Scatter(x=final_df['MONTH'], y=final_df['COUNT'],
                        mode='lines+markers',showlegend=True, name=type))

    #craete LineChart figure data
    figure = {
        'data':traces,
        'layout': {
            'title': 'Crime Trends Data Visualization',
            'xaxis': dict(
                title="Months",
                titlefont=dict(
                    family='Helvetica, monospace',
                    color='#7f7f7f'
                )),
            'yaxis': dict(
                title="Crime Count",
                titlefont=dict(
                    family='Helvetica, monospace',
                    color='#7f7f7f'
                )),
        }
    }
    return figure

def getSubfilterType(str):
    types = []
    if "Theft" in str:
        types.append('Other Theft')
        types.append('Theft from Vehicle')
        types.append('Theft of Vehicle')
        types.append('Theft of Bicycle')
        return types

    elif str == "Break In":
        types.append("Break and Enter Commercial")
        types.append("Break and Enter Residential/Other")
        return types

    elif str == "Mischief":
        types.append("Mischief")
        return types

    elif str == "Accident":
        types.append("Vehicle Collision or Pedestrian Struck (with Injury)")
        types.append("Vehicle Collision or Pedestrian Struck (with Fatality)")
        return types

    else:
        return types

def getSubTypeFilteredDf(subTypes, yr_filtered_df):
    df = yr_filtered_df
    if "Mischief" == subTypes[0]:
        df = yr_filtered_df[yr_filtered_df.TYPE == 'Mischief']
        return df
    elif "Theft" in subTypes[0]:
        df = yr_filtered_df[yr_filtered_df.TYPE.astype(str).str.contains("Theft")]
        return df
    elif "Break" in subTypes[0]:
        df = yr_filtered_df[yr_filtered_df.TYPE.isin(subTypes)]
        return df
    elif "Fatality" in subTypes[0]:
        df = yr_filtered_df[yr_filtered_df.TYPE.astype(str).str.contains("Fatality")]
    else:
        return df

# This function give BarChart data based on NEIGHBOURHOOD and selected year, 2017 is default year
def getBarChartData(selected_year):

    yr_filtered_df = df_crime[df_crime['YEAR'] == int(selected_year)]
    n_df = yr_filtered_df.groupby('NEIGHBOURHOOD').count()
    se = (n_df.TYPE)
    nc_final_df = pd.DataFrame({'NEIGHBOURHOOD': se.index, 'COUNT': se.values})
    title="Crime in "+ str(selected_year);
    figure=(px.bar(nc_final_df, y='NEIGHBOURHOOD', x='COUNT', orientation='h',
             hover_data=['NEIGHBOURHOOD', 'COUNT'], color='COUNT',title=title))

    return figure

# This function give PieChartdata contain Crime percentage in different street of neighbourhhod based on selected neighbourhood from BarChart.
def getCrimePieChartDataFunction(selected_year,neighbourhood):

    yr_filtered_df = df_crime[df_crime['YEAR'] == int(main_filter_yr)]
    nh_filtered_df= yr_filtered_df[yr_filtered_df['NEIGHBOURHOOD'] == str(neighbourhood)]
    streets = nh_filtered_df.HUNDRED_BLOCK
    stlist = []
    for st in streets:
        wordlist = st.split(" ")
        newwordlist = []
        for word in wordlist:
            if any(ch.isdigit() for ch in word):
                continue
            elif "XX" in word:
                continue
            elif len(word) == 1 and "/" not in word:
                continue
            elif "BLOCK" in word:
                continue
            else:
                newwordlist.append(word)
        newstname = " ".join(newwordlist)
        if "/" in newstname:
            newstname = "UNKNOWN"
        stlist.append(newstname)

    nh_filtered_df.HUNDRED_BLOCK = stlist
    st_grouped_df =nh_filtered_df.groupby('HUNDRED_BLOCK').count()
    se = (st_grouped_df.TYPE)
    pc_final_df = pd.DataFrame({'STREET': se.index, 'COUNT': se.values})
    ctlist = pc_final_df.COUNT
    total = pc_final_df.COUNT.sum()
    perlist = []
    for c in ctlist:
        per = (c / total) * 100
        perlist.append(per)

    pc_final_df["Percentage"] = perlist
    streetlist = []
    for index, row in pc_final_df.iterrows():
        if (row['Percentage'] < 2.0):
            row['STREET'] = "OTHERS"
        streetlist.append(row['STREET'])

    pc_final_df.STREET = streetlist

    labels = pc_final_df.STREET
    values = pc_final_df.COUNT
    title = "Street-wise Crime Visualization for " + neighbourhood

    # craete pieChart figure data
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, title_text=title)])
    return fig

#get PieChartData with no of crime based on selected neighbourHood in barchart.
def getPieChartDataFunction(selected_year,neighbourhood):

    yr_filtered_df = df_crime[df_crime['YEAR'] == int(selected_year)]
    nh_filtered_df = yr_filtered_df[yr_filtered_df['NEIGHBOURHOOD'] == str(neighbourhood)]
    hrs = nh_filtered_df.HOUR
    dayparts = []

    for hr in hrs:
        if hr >= 0.0 and hr < 6.0:
            daypart = "Overnight (12pm-6am)"
        elif hr >= 6.0 and hr < 10.0:
            daypart = "Morning drive (6am-10am)"
        elif (hr >= 10.0 and hr < 15.0):
            daypart = "Daytime (10am-3pm)"
        elif (hr >= 15.0 and hr < 19.0):
            daypart = "Afternoon drive (3pm-7pm)"
        elif (hr >= 19.0 and hr < 24.0):
            daypart = "Nighttime (7pm-12pm)"
        dayparts.append(daypart)

    nh_filtered_df['DAYSLOT'] = dayparts
    time_grouped_df = nh_filtered_df.groupby('DAYSLOT').count()
    se = (time_grouped_df.TYPE)
    tpc_final_df = pd.DataFrame({'DAYSLOT': se.index, 'COUNT': se.values})
    labels = tpc_final_df.DAYSLOT
    values = tpc_final_df.COUNT
    title = "Time-wise Crime Visualization for " + neighbourhood
    fig = go.Figure(data=[go.Pie(labels=labels, values=values,title_text=title)])
    return fig

# This Url called when user click on specific Neighbourhood in barChart, Crime list  based on time.
@app.route("/getPieGraph",methods=['PUT','GET'])
def getPieChartFunction():
    selectedYear = request.args['selectedYear'];
    selectedNeighbourhood = request.args['selectedNeighbourhood'];
    timePieChart_data=getPieChartDataFunction(selectedYear,selectedNeighbourhood);
    timePieChart_data = json.dumps(timePieChart_data, cls=plotly.utils.PlotlyJSONEncoder)
    return timePieChart_data

#This url Called when user click specific neighbourhood in barchart, Crime list  based on Street in neighbourhood.
@app.route("/getCrimePieGraph",methods=['PUT','GET'])
def getCrimePieChartFunction():

    selectedYear = request.args['selectedYear'];
    selectedNeighbourhood = request.args['selectedNeighbourhood'];
    crime_pieChartData = getCrimePieChartDataFunction(selectedYear, selectedNeighbourhood);
    crime_pieChartData = json.dumps(crime_pieChartData, cls=plotly.utils.PlotlyJSONEncoder)
    return crime_pieChartData

#This Url Called when user select crime by neighbourHood from frontend.
@app.route('/barChartFunction')
def barChartFunction():
    figure_data=getBarChartData(2017);
    data = json.dumps(figure_data, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("crimeByNeighbourhood.html",data=data)

#This Url called when user select year to see Neighbour with crime counts based on year.
@app.route('/getBarChart')
def getBarChart():
    selectedYear = request.args['selectedYear'];
    figure_data=getBarChartData(selectedYear);
    data = json.dumps(figure_data, cls=plotly.utils.PlotlyJSONEncoder)
    return data

#This function Create baseMap for HeatMap
def generateBaseMap(default_location=[49.2827, -123.1207], default_zoom_start=12):
    map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return map

#This url return HeatMAp data
@app.route('/getHeatMap')
def getHeatMap():

    df = pd.read_csv('crime.csv')
    crimeTypeList=[];

    #default Theft type crime and 2017 year
    crimeTypeList.append("Theft");
    df = getSelectedCrimeData(crimeTypeList, 2017, df);
    df['count'] = 1
    heatMap = generateBaseMap()
    HeatMap(data=df[['Latitude', 'Longitude', 'count']].groupby(
        ['Latitude', 'Longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(heatMap)
    heatMap.save('static/heatMap.html')
    return render_template("heatMap.html")

#This Url called when user clicked GetModel Data.
def predict_crime():
    df_crime = pd.read_csv('crime.csv')

    # Removing homocide and offense against a person as they details about these incidents are not available.
    df_crime = df_crime[df_crime.TYPE != 'Homicide']
    df_crime = df_crime[df_crime.TYPE != 'Offence Against a Person']

    # Dropiing the columns which are not required.
    df_crime.drop(['HUNDRED_BLOCK', 'X', 'Y', 'Latitude', 'Longitude'], axis=1, inplace=True)
    df_crime['DATE'] = pd.to_datetime(df_crime.YEAR * 10000 + df_crime.MONTH * 100 + df_crime.DAY, format='%Y%m%d')

    # Creating Day_type column as weekend or weekday
    df_crime['DAY_TYPE'] = ((pd.DatetimeIndex(df_crime.DATE).dayofweek) // 5 == 1).astype(float)
    df_crime['DAY_TYPE'] = df_crime['DAY_TYPE'].astype('str')
    df_crime['DAY_TYPE'].replace('0.0', 'WEEKDAY', inplace=True)
    df_crime['DAY_TYPE'].replace('1.0', 'WEEKEND', inplace=True)

    df_crime['TYPE'].replace('Break and Enter Residential/Other', 'Break and Enter', inplace=True)
    df_crime['TYPE'].replace('Break and Enter Commercial', 'Break and Enter', inplace=True)
    df_crime['TYPE'].replace('Other Theft', 'Theft', inplace=True)
    df_crime['TYPE'].replace('Theft from Vehicle', 'Theft', inplace=True)
    df_crime['TYPE'].replace('Theft of Bicycle', 'Theft', inplace=True)
    df_crime['TYPE'].replace('Theft of Vehicle', 'Theft', inplace=True)
    df_crime['TYPE'].replace('Vehicle Collision or Pedestrian Struck (with Fatality)', 'Vehicle Collision',
                             inplace=True)
    df_crime['TYPE'].replace('Vehicle Collision or Pedestrian Struck (with Injury)', 'Vehicle Collision', inplace=True)

    # Taking 2017 year for predictions
    df_final_plot = df_crime[df_crime.YEAR == 2017].reset_index(drop=True)
    df_crime = df_crime[df_crime.YEAR != 2017]

    # Creating the features
    features = df_crime[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'NEIGHBOURHOOD', 'DAY_TYPE']].reset_index(drop=True)

    # creating the target columns
    target = df_crime[['TYPE']].reset_index(drop=True)

    # Encoding the features
    features['NEIGHBOURHOOD'] = features['NEIGHBOURHOOD'].astype('category')
    features = pd.get_dummies(features, columns=['NEIGHBOURHOOD'], prefix=['NEIGHBOURHOOD'])

    features['DAY_TYPE'].replace('WEEKDAY', '0', inplace=True)
    features['DAY_TYPE'].replace('WEEKEND', '1', inplace=True)
    features['DAY_TYPE'] = features['DAY_TYPE'].astype('int')

    # Scaling the features
    train_norm = features[['YEAR', 'MONTH', 'DAY', 'MINUTE']]
    std_scale = preprocessing.StandardScaler().fit(train_norm)
    x_train_norm = std_scale.transform(train_norm)
    training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
    normalized_data = features.update(training_norm_col)
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'species'.
    target['TYPE'] = label_encoder.fit_transform(target['TYPE'])
    target['TYPE'].unique()

    # Creating the hold out set evaluation.
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Predicting the crime type using random forest
    rfClf = RandomForestClassifier(n_estimators=10)
    rfClf.fit(X_train, y_train.values.ravel())
    y_pred_training_rf = rfClf.predict(X_train)
    y_pred_validation_rf = rfClf.predict(X_test)

    # Measuring accuracy and f1 score
    accuracy_training_RF = accuracy_score(y_train, y_pred_training_rf)
    f1_score_training_RF = f1_score(y_train, y_pred_training_rf, average='micro')

    accuracy_validation_RF = accuracy_score(y_test, y_pred_validation_rf)
    f1_score_validation_RF = f1_score(y_test, y_pred_validation_rf, average='micro')

    # Predicting with logistic regression to show the difference between models using LIME.
    log_reg = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    y_pred_trainig_lg = log_reg.predict(X_train)
    y_pred_validation_lg = log_reg.predict(X_test)

    accuracy_training_logReg = accuracy_score(y_train, y_pred_trainig_lg)

    f1_score_training_logReg = f1_score(y_train, y_pred_trainig_lg, average='micro')
    accuracy_validation_logReg = accuracy_score(y_test, y_pred_validation_lg)
    f1_score_validation_logReg = f1_score(y_test, y_pred_validation_lg, average='micro')

    # Showing the model Interpretability using LIME.
    my_sample1 = np.array(X_test.iloc[1]).reshape(1, -1)
    my_sample2 = np.array(X_test.iloc[2]).reshape(1, -1)


    # Creating a feature name list for LIME
    feature_names = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'DAY_TYPE',
                     'NEIGHBOURHOOD_Arbutus Ridge',
                     'NEIGHBOURHOOD_Central Business District',
                     'NEIGHBOURHOOD_Dunbar-Southlands', 'NEIGHBOURHOOD_Fairview',
                     'NEIGHBOURHOOD_Grandview-Woodland', 'NEIGHBOURHOOD_Hastings-Sunrise',
                     'NEIGHBOURHOOD_Kensington-Cedar Cottage', 'NEIGHBOURHOOD_Kerrisdale',
                     'NEIGHBOURHOOD_Killarney', 'NEIGHBOURHOOD_Kitsilano',
                     'NEIGHBOURHOOD_Marpole', 'NEIGHBOURHOOD_Mount Pleasant',
                     'NEIGHBOURHOOD_Musqueam', 'NEIGHBOURHOOD_Oakridge',
                     'NEIGHBOURHOOD_Renfrew-Collingwood', 'NEIGHBOURHOOD_Riley Park',
                     'NEIGHBOURHOOD_Shaughnessy', 'NEIGHBOURHOOD_South Cambie',
                     'NEIGHBOURHOOD_Stanley Park', 'NEIGHBOURHOOD_Strathcona',
                     'NEIGHBOURHOOD_Sunset', 'NEIGHBOURHOOD_Victoria-Fraserview',
                     'NEIGHBOURHOOD_West End', 'NEIGHBOURHOOD_West Point Grey']

    # Creating class_names for LIME
    class_names = target['TYPE'].unique()

    # The explainer for LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names,
                                                       class_names=class_names, discretize_continuous=True)

    # Model interpretability with LIME for  Sample 1 with logistic regression and random forest
    rfClf.predict(my_sample1)
    exp1_rf = explainer.explain_instance(X_test.iloc[1], rfClf.predict_proba, num_features=30, top_labels=1)
    exp1_rf.save_to_file('static/lime_instance1_rf.html')

    log_reg.predict(my_sample1)
    exp1_logReg = explainer.explain_instance(X_test.iloc[1], log_reg.predict_proba, num_features=30, top_labels=1)
    exp1_logReg.save_to_file('static/lime_instance1_logReg.html')

    # Model interpretability with LIME for  Sample 2 with logistic regression and random forest
    rfClf.predict(my_sample2)
    exp2_rf = explainer.explain_instance(X_test.iloc[2], rfClf.predict_proba, num_features=30, top_labels=1)
    exp2_rf.save_to_file('static/lime_instance2_rf.html')

    log_reg.predict(my_sample2)
    exp1 = explainer.explain_instance(X_test.iloc[2], log_reg.predict_proba, num_features=30, top_labels=1)
    exp1.save_to_file('static/lime_instance2_logReg.html')

    # Saving the output of model interpretability as HTML.

    # Predicting on second month of 2017
    df_final_plot = df_final_plot[df_final_plot.MONTH == 3].reset_index(drop=True)
    df_final_plot.drop(['TYPE'], axis=1, inplace=True)

    # Repeating same procedure for this selection of month
    df_predict = df_final_plot[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'NEIGHBOURHOOD', 'DAY_TYPE']].reset_index(
        drop=True)
    df_predict['NEIGHBOURHOOD'] = df_predict['NEIGHBOURHOOD'].astype('category')
    df_predict = pd.get_dummies(df_predict, columns=['NEIGHBOURHOOD'], prefix=['NEIGHBOURHOOD'])
    df_predict['DAY_TYPE'].replace('WEEKDAY', '0', inplace=True)
    df_predict['DAY_TYPE'].replace('WEEKEND', '1', inplace=True)
    df_predict['DAY_TYPE'] = features['DAY_TYPE'].astype('int')

    train_norm = df_predict[['YEAR', 'MONTH', 'DAY', 'MINUTE']]
    std_scale = preprocessing.StandardScaler().fit(train_norm)
    x_train_norm = std_scale.transform(train_norm)
    training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
    normalized_data = df_predict.update(training_norm_col)
    y_pred_final = rfClf.predict(df_predict)
    df_final_plot['TYPE'] = pd.DataFrame(y_pred_final)
    df_final_plot['TYPE'].value_counts()
    df_final_plot = df_final_plot[np.isfinite(df_final_plot['TYPE'])]
    df_final_plot['TYPE'] = df_final_plot['TYPE'].astype('str')
    df_final_plot['TYPE'].replace('2', 'THEFT', inplace=True)
    df_final_plot['TYPE'].replace('0', 'MISCHIEF', inplace=True)
    df_final_plot['TYPE'].replace('1', 'BREAK And ENTER', inplace=True)
    df_final_plot['TYPE'].replace('3', 'VEHICLE COLLISION', inplace=True)

    return df_final_plot


def plot_model():
    global model_data
    # Getting the predicted data frame from predict crime method.
    model_data = predict_crime()
    df_final_plot = model_data

    # Creaing labels for shanky diagram
    df_final_plot['TYPE_NO'] = df_final_plot['TYPE']
    df_final_plot['TYPE_NO'].replace('THEFT', '2', inplace=True)
    df_final_plot['TYPE_NO'].replace('MISCHIEF', '0', inplace=True)
    df_final_plot['TYPE_NO'].replace('BREAK And ENTER', '1', inplace=True)
    df_final_plot['TYPE_NO'].replace('VEHICLE COLLISION', '3', inplace=True)
    df_final_plot['TYPE_NO'] = df_final_plot['TYPE_NO'].astype('int32')

    NEIGHBOURHOOD = go.parcats.Dimension(
        values=df_final_plot.NEIGHBOURHOOD, label="NEIGHBOURHOOD")

    DAY_TYPE = go.parcats.Dimension(values=df_final_plot.DAY_TYPE, label="DAY_TYPE")
    TYPE = go.parcats.Dimension(values=df_final_plot.TYPE, label="CRIME TYPE")
    color = df_final_plot.TYPE_NO;

    # Creating the shanky diagram with plotly go.
    fig = go.Figure(data=[go.Parcats(dimensions=[NEIGHBOURHOOD, DAY_TYPE, TYPE],
                                     line={'color': color, 'colorscale': 'rdbu'},
                                     labelfont={'size': 12, 'family': 'Times'},
                                     tickfont={'size': 10, 'family': 'Times'},
                                     arrangement='freeform')])

    return fig

#Url to render to html page which contain Model information..
@app.route('/getModel')
def getModel():
    return render_template("modelPage.html")

#Url to render Page which contain Model Information.
@app.route('/getModelData')
def getModelData():
    fig_data=plot_model()
    data = json.dumps(fig_data, cls=plotly.utils.PlotlyJSONEncoder)
    return data

#This function give ParallelGraph.
@app.route('/getParallelGraph')
def plot_parallel_coordinates():
    data=plot_parallel_coordinates();
    data = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return data

#This function Give InterpretationModel with theft crime type data
@app.route('/InterpretationModel1')
def InterpretationModel1():
    return render_template("limeGraph1.html")

#This function Give InterpretationModel with mischieft  crime type datadata
@app.route('/InterpretationModel2')
def InterpretationModel2():
    return render_template("limeGraph2.html")

#This function give figure data for parallel model visualization model.
def plot_parallel_coordinates():
    # Getting the predicting data frame which is returned from predict_crime method.
    df_final_plot = model_data;
    df_final_plot['DAY_TYPE'].replace('WEEKDAY', '0', inplace=True)
    df_final_plot['DAY_TYPE'].replace('WEEKEND', '1', inplace=True)
    df_final_plot['DAY_TYPE'] = df_final_plot['DAY_TYPE'].astype('int32')

    df_final_plot['TYPE'].replace('THEFT', '2', inplace=True)
    df_final_plot['TYPE'].replace('MISCHIEF', '0', inplace=True)
    df_final_plot['TYPE'].replace('BREAK And ENTER', '1', inplace=True)
    df_final_plot['TYPE'].replace('VEHICLE COLLISION', '3', inplace=True)
    df_final_plot['TYPE'] = df_final_plot['TYPE'].astype('int32')

    df_final_plot['NEIGHBOURHOOD'].replace('Musqueam', '0', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Stanley Park', '1', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('West Point Grey', '2', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('South Cambie', '3', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Kerrisdale', '4', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Dunbar-Southlands', '5', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Arbutus Ridge', '6', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Shaughnessy', '7', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Killarney', '8', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Oakridge', '9', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Victoria-Fraserview', '10', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Riley Park', '11', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Sunset', '12', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Marpole', '13', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Kensington-Cedar Cottage', '14', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Kitsilano', '15', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Hastings-Sunrise', '16', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Fairview', '17', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Renfrew-Collingwood', '18', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Strathcona', '19', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('randview-Woodland', '20', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Mount Pleasant', '21', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('West End', '22', inplace=True)
    df_final_plot['NEIGHBOURHOOD'].replace('Central Business District', '23', inplace=True)
    df_final_plot['TYPE'] = df_final_plot['TYPE'].astype('int32')



    # Creating the parallel coordinates with plotly.
    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=df_final_plot['TYPE'],
                  colorscale=[[0.0, "rgb(165,0,38)"],
                              [0.1111111111111111, "rgb(215,48,39)"],
                              [0.2222222222222222, "rgb(244,109,67)"],
                              [0.3333333333333333, "rgb(253,174,97)"],
                              [0.4444444444444444, "rgb(254,224,144)"],
                              [0.5555555555555556, "rgb(224,243,248)"],
                              [0.6666666666666666, "rgb(171,217,233)"],
                              [0.7777777777777778, "rgb(116,173,209)"],
                              [0.8888888888888888, "rgb(69,117,180)"],
                              [1.0, "rgb(49,54,149)"]],
                  showscale=True),
        dimensions=list([
            dict(range=[0, 12],
                 label='MONTH', values=df_final_plot['MONTH']),
            dict(range=[0, 31],
                 label='DAYS', values=df_final_plot['DAY']),
            dict(range=[0, 24],
                 label='HOURS', values=df_final_plot['HOUR']),
            dict(range=[0, 60],
                 label='MINUTES', values=df_final_plot['MINUTE']),
            dict(range=[-1, 2],
                 tickvals=[0, 1],
                 ticktext=['WEEKDAY', 'WEEKEND'],
                 label='DAY TYPE', values=df_final_plot['DAY_TYPE']),
            dict(range=[0, 24],
                 tickvals=[0, 5, 10, 15, 20, 23],
                 ticktext=['Musqueam', 'Dunbar-Southlands', 'Victoria-Fraserview', 'Kitsilano', 'randview-Woodland',
                           'CBP'],
                 label='NEIGHBOURHOOD', values=df_final_plot['NEIGHBOURHOOD']),
            dict(tickvals=[0, 1, 2, 3],
                 ticktext=['MISCHIEF', 'BREAK And ENTER', 'THEFT', 'VEHICLE COLLISION'],
                 label='CRIME TYPE', values=df_final_plot['TYPE'])
        ])
    )
    )
    return fig

@app.route('/homePage')
def homePage():

    df = pd.read_csv('crime.csv')
    CrimeTypeList=["Theft"];
    subset_of_df = getSelectedCrimeData(CrimeTypeList,2017,df)
    getMap(subset_of_df)
    graphData = getLineChartData("2017", CrimeTypeList)
    graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("homePage.html", data=graphJSON)

# This function give selected dataframe based on crimeType and year.
def getSelectedCrimeData(CrimeTypeList, year, df):

    df = df[df['YEAR'].apply(lambda X: str(X) == str(year))]

    crimeList=[]

    for i in CrimeTypeList:
        if i=="Theft":
            crimeList.append("other Theft");
            crimeList.append("Theft fromVehicle");
            crimeList.append("Theft of Bicycle");
            crimeList.append("Theft of Vehicle");
        elif i=="Accident":
            crimeList.append("Vehicle Collision or Pedestrian Struck (with Fatality)");
            crimeList.append("Vehicle Collision or Pedestrian Struck (with Injury)");
        elif i=="Break In":
            crimeList.append("Break and Enter Commercial");
            crimeList.append("Break and Enter Residential/Other");
        elif i=="Mischief":
            crimeList.append("Mischief");
    df = df[df['TYPE'].apply(lambda X: str(X) in crimeList)];

    return df

@app.route('/getHeatMapData')
def getHeatMapData():
    df = pd.read_csv('crime.csv')
    selectedYear = request.args['selectedYear'];
    selectedCrime = ast.literal_eval(request.args['selectedCrime']);
    df = getSelectedCrimeData(selectedCrime, selectedYear, df);
    df['count'] = 1

    #generate BaseMap
    heatMap = generateBaseMap()

    #generate Heatmap
    HeatMap(data=df[['Latitude', 'Longitude', 'count']].groupby(
            ['Latitude', 'Longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(heatMap)

    #save to Html
    heatMap.save('static/heatMap.html')
    chart_data = df.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return data

if __name__ == "__main__":
    app.run(debug=True)
