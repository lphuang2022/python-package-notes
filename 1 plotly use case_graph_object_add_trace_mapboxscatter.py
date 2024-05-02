
# -------------------------
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import utils.data_FIR as data_FIR
import utils.boundaryFIR as boundaryFIR
from utils.utils_funcs import arrivalAC_FPL, gen_wp2wsssDist


 
wsss = {'lon':103.988, 
        'lat':1.354}

#------------------ load FPL for arrival a/c filter
dir_data = "./data"
csv_FPL = "Flight Plan_20220901-20221130.csv"
df_FPL = pd.read_csv(f"{dir_data}/{csv_FPL}")
print(df_FPL.head(10))

# near airports
dist_nearAirports = 800 # nm
'''
within 800 nm: 28 airports
within 700 nm: 22
within 600 nm: 19
within 500 nm: 14
within 400 nm: 13
within 300 nm: 7
within 200 nm: 6
within 100 nm: 2
'''
csv_airports = "airports.csv"
df_airports = pd.read_csv(f"{dir_data}/{csv_airports}")
airports2wsss = df_FPL[df_FPL['ADES']=='WSSS']['ADEP'].unique().tolist()
df_airports_near = df_airports[(df_airports['dist2wsss_nm']< dist_nearAirports) & (df_airports['country']!='Singapore') & df_airports['ICAO'].isin(airports2wsss)].reset_index(drop=True)
df_airports_near['dist2wsss_nm'] = df_airports_near['dist2wsss_nm'].astype(str)
df_airports_near['dist2wsss_nm'] = df_airports_near['dist2wsss_nm'].apply(lambda x: ' ' + x.split('.')[0])
df_airports_near['ICAO(dist2wsss)'] = df_airports_near['ICAO'] + df_airports_near['dist2wsss_nm']
print(f"No. of near airports (<{dist_nearAirports} nm): {len(df_airports_near['ICAO'].unique().tolist())}")

#----------------- load and process FIR boundary---------------
points_FIR = boundaryFIR.gen_boundaryPoints_FIR()
lats_FIR, lons_FIR = zip(*points_FIR)


# generate points for airspace partition
import collections
radius_km = 400/0.53996
def gen_partition_points():
    dict_points_lat = collections.defaultdict(float)
    dict_points_lon = collections.defaultdict(float)
    for theta in range(0, 360, 10):
        alpha=theta*np.pi/180
        point_lon = data_FIR.coor_wsss[1] + (radius_km / 111.32) * np.cos(alpha)
        point_lat = data_FIR.coor_wsss[0] + (radius_km / 111.32) * np.sin(alpha)
        dict_points_lat[alpha] = point_lat
        dict_points_lon[alpha] = point_lon
       

    return  dict_points_lat,  dict_points_lon

dict_points_lat,  dict_points_lon = gen_partition_points()


# ----------------load waypoints and process--------------
dir_wp = "./data"
file_wp = "wp_star"
df_wp = pd.read_csv(f"{dir_wp}/{file_wp}.csv")
df_wp['lat'] = df_wp['Latitude '].apply(lambda x: x.strip()[:-1])
df_wp['lat'] = df_wp['lat'].apply(lambda x: float(x[:2]) + float(x[2:4])/60 + float(x[4:])/3600)
df_wp['lon'] = df_wp['Longitude '].apply(lambda x: x.strip()[:-1])
df_wp['lon'] = df_wp['lon'].apply(lambda x: float(x[:3]) + float(x[3:5])/60 + float(x[5:])/3600)
df_wp_star = df_wp[df_wp['TMA_STAR']=='STAR'].reset_index(drop=True)
df_wp_star['wp'] = df_wp_star['Name ']
df_wp_star = gen_wp2wsssDist(df_wp_star)#column 'dist2wsss_nm'
df_wp_star['dist2wsss_nm'] = df_wp_star['dist2wsss_nm'].astype(str)
df_wp_star['dist2wsss_nm'] = df_wp_star['dist2wsss_nm'].apply(lambda x: ' ' + x.split('.')[0])
df_wp_star['wp(dist2wsss)'] = df_wp_star['wp'] + df_wp_star['dist2wsss_nm']
# df_wp_star.to_csv(f"{dir_wp}/wp2wsssDist.csv")

wp_hld = ['BOBAG ', 'ELALO ', 'IKIMA ', 'KARTO ', 'LAVAX ', 'MABAL ', 'NYLON ', 'REMES ', 'REPOV ', 'SAMKO ', 'PIBAP ', 'PASPU ']

df_wp_star_hld = df_wp_star[df_wp_star['wp'].isin(wp_hld)].reset_index(drop=True)
df_wp_star = df_wp_star[~df_wp_star['wp'].isin(wp_hld)].reset_index(drop=True)


#----------------- load and process track---------------
dir1 = "C:/_Huang/data22Analyze/analyze_CAT21/data_CAT21/csv_cat21"
dir = "C:/_Huang/data22Analyze/analyze_CAT21/CAT21_noFilter/csv"
dir2 =  "D:/atop2/CAT21_2022CSV"
date = 221025
df = pd.DataFrame()
csv = f"{date}.csv"
file = f"{dir2}/{csv}"
df = pd.read_csv(file, index_col=0)
df = df.drop_duplicates(subset=['lat', 'lon'])

# label hour
df['dt'] = pd.to_datetime(df['timeH'], unit='s')
format = "%m%d %H:%M:%S"
df['time_utc'] = df['dt'].dt.strftime(format)
df['hour'] = df['dt'].dt.hour
df['day'] = df['dt'].dt.day

# ------ # ----------filter: arrival A/C-----
callsign=""

# filter callsign: comment two lines below
# callsign = "AIC346"
# df = df[df['callsign']==callsign]
# callsign = ['SIA910', 'SIA916']
# callsign = ['SIA319']
# df = df[df['callsign'].isin(callsign)]

arrivals = arrivalAC_FPL(df_FPL, date)
df = df[df['callsign'].isin(arrivals)].reset_index(drop=True)

# -------------------filter: hours-----------
hours = list(range(8,20))
df = df[df['hour'].isin(hours)].reset_index(drop=True)

# dir_arrival = "C:/_Huang/ATOP2/data22"
# dir_arrival =  "D:/data22Analyze/1_itsc_paperRevise/csv_eta"
# file_arrival = f"{date}_slot_60_sta_10"
# df_arr = pd.read_csv(f"{dir_arrival}/{file_arrival}.csv")
# callsign_arrival = df_arr['callsign'].tolist()
# df = df[df['callsign'].isin(callsign_arrival)].reset_index(drop=True)

df = df.drop_duplicates(subset=['callsign', 'timeH'], keep='first').reset_index(drop=True)
df = df.sort_values(by=['callsign','timeH']).reset_index(drop=True)
df = df[['timeH', 'lat', 'lon', 'callsign']]




space_angRange = [space_angleRange_400To200nm]  # List of dictionaries
combined_dict = {k: v for d in space_angRange for k, v in d.items()}



import collections
dict_seg_lats = collections.defaultdict(list)
dict_seg_lons = collections.defaultdict(list)
dist_ring = [400, 200, 50, 10]
for i in range(len(dist_ring)-1):
    r1, r0 = dist_ring[i]/0.53996, dist_ring[i+1]/0.53996
    for _, vals in combined_dict.items():
        for beta in vals:
            for theta in beta:
                if theta == 180 or theta == -180: continue
                alpha=theta*np.pi/180
                point_lon1 = data_FIR.coor_wsss[1] + (r1 / 111.32) * np.cos(alpha)
                point_lat1 = data_FIR.coor_wsss[0] + (r1 / 111.32) * np.sin(alpha)

                point_lon0 = data_FIR.coor_wsss[1] + (r0 / 111.32) * np.cos(alpha)
                point_lat0 = data_FIR.coor_wsss[0] + (r0 / 111.32) * np.sin(alpha)
                
                dict_seg_lats[(dist_ring[i], dist_ring[i+1], theta)] = [point_lat1, point_lat0]
                dict_seg_lons[(dist_ring[i], dist_ring[i+1], theta)] = [point_lon1, point_lon0]
                

           

dir_data =  "D:\\atop2\\process4AMAN\\data\\data_tabFeatures_v3.2"
data = pd.read_csv(f'{dir_data}/{date}_eta_tabFeatures_v32.csv', index_col=0)
data = data[data['dist2wsss']<=400].reset_index(drop=True)

data = data[['callsign', 'lat', 'lon', 'dist2wsss', 'airspace_label', 'eta']]
data['ring'] = data['dist2wsss']//50
data = data.sort_values(by=['callsign', 'eta'], ascending=False).reset_index(drop=True)

data_eta_max = data.groupby(['airspace_label', 'ring', 'callsign'], as_index=False)['eta'].max()
data_eta_min = data.groupby(['airspace_label', 'ring', 'callsign'], as_index=False)['eta'].min()
data_eta_max['eta_enter'] = data_eta_max['eta']
data_eta_min['eta_exit'] = data_eta_min['eta']
data_travelTime = pd.merge(data_eta_max, data_eta_min, on=['airspace_label', 'ring', 'callsign'])
data_travelTime.drop(columns=['eta_x', 'eta_y'], inplace=True)
data_travelTime['travelTime'] = (data_travelTime['eta_enter'] - data_travelTime['eta_exit'])/60

data_lat_lon_mean = data[['airspace_label', 'ring', 'lat', 'lon']].groupby(['airspace_label', 'ring'], as_index=False)[['lat', 'lon']].mean()
data_travelTime_mean = data_travelTime.groupby(['airspace_label', 'ring'], as_index=False)['travelTime'].mean()
data_travelTime_std = data_travelTime.groupby(['airspace_label', 'ring'], as_index=False)['travelTime'].std()

df_eta = pd.merge(data_lat_lon_mean, data_travelTime_mean)
# df_eta = pd.merge(data_lat_lon_mean, data_travelTime_std)
df_eta = df_eta.dropna().reset_index(drop=True)

df_eta['travelTime'] = round(df_eta['travelTime'], 1)
df_eta['travelTime_str'] = df_eta['travelTime'].astype(str)

FIR_plot = False
# ------------------------------------------  plot    ---------------------------------------
if not FIR_plot:
    fig = go.Figure()
    
else:
# FIR
    fig = go.Figure(go.Scattermapbox(
        mode="lines",
        fill="toself",  # Fill the area enclosed by the points
        lon=lons_FIR,
        lat=lats_FIR,
        marker={'size': 1, 'color': "aquamarine", 'opacity':0.1}  # Customize marker appearance
    ))

   


# -----------------------------0. add circles-----------------

# Generate points for the circle
for r in range(400, 49, -50):
    radius_km = r/0.53996
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle_lon = data_FIR.coor_wsss[1] + (radius_km / 111.32) * np.cos(theta)
    circle_lat = data_FIR.coor_wsss[0] + (radius_km / 111.32) * np.sin(theta)

    fig.add_trace(go.Scattermapbox(
        lat=circle_lat,
        lon=circle_lon,
        mode='lines',
        marker={'size': 1, 'color': "pink", 'opacity':0.1}
        # line=dict(width=2, color='blue'),
        # fill='toself',  # Fill the circle
        # fillcolor='rgba(0, 0, 255, 0.2)',  # Light blue fill color
    ))

for key in dict_seg_lons.keys():
    line_lats = np.array(dict_seg_lats[key])
    line_lons = np.array(dict_seg_lons[key])
    fig.add_trace(go.Scattermapbox(
        lat=line_lats,
        lon=line_lons,
        mode='lines',
        marker={'size': 3, 'color': "pink", 'opacity':0.1}
    ))

# for alpha in dict_points_lon.keys():
#     line_lats = np.array([data_FIR.coor_wsss[0], dict_points_lat[alpha]])
#     line_lons = np.array([data_FIR.coor_wsss[1], dict_points_lon[alpha]])
#     fig.add_trace(go.Scattermapbox(
#         lat=line_lats,
#         lon=line_lons,
#         mode='lines',
#         marker={'size': 10, 'color': "orange", 'opacity':0.1}
#     ))

# ------------------------ 1. track ----------------------
show_callsign = False
fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers' if not show_callsign else 'markers+text',
        # mode='markers+text',
        # mode = 'markers',
        marker=go.scattermapbox.Marker(
            size=4,
            color='blue',
            opacity=0.7
        ),
        text=list(df['callsign']),
        hoverinfo='none'
    ))

# -----------------------------2. add wp-----------------
fig.add_trace(go.Scattermapbox(
        lat=df_wp_star['lat'],
        lon=df_wp_star['lon'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=30,
            color='white',
            opacity=0.3
        ),
        text=list(df_wp_star['wp(dist2wsss)']),
        hoverinfo='text'
    ))

fig.add_trace(go.Scattermapbox(
        lat=df_wp_star_hld['lat'],
        lon=df_wp_star_hld['lon'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=20,
            color='pink',
            opacity=0.7,
        ),
        text=list(df_wp_star_hld['wp(dist2wsss)']),
        hoverinfo='text'
    ))


#----------------------------3. near airports--------------
show_nearAirports = True
fig.add_trace(go.Scattermapbox(
        lat=df_airports_near['lat'],
        lon=df_airports_near['lon'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=50,
            color='rgb(120, 120, 0)',
            opacity=0.6
        ),
        text=list(df_airports_near['ICAO(dist2wsss)']),
        hoverinfo='text'
    ))

fig.update_traces(textfont=dict(color="white", size=16), selector=dict(mode="markers"))

#  -----------------------  4. travel time  --------------
fig.add_trace(go.Scattermapbox(
    lon=df_eta['lon'],
    lat=df_eta['lat'],
    text=df_eta['travelTime_str'],
    mode='markers+text',
    # marker_color=df['travelTime'],
    marker=go.scattermapbox.Marker(
            size=5+df_eta['travelTime']*3,
            color=(df_eta['travelTime']+300)/255,
            opacity=0.9
        ),
    hoverinfo='text'
))


# mapbox
fig.update_layout(
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(bearing=0,
                center=dict(lat=wsss['lat'],lon=wsss['lon']),
                pitch=0,
                zoom=3,
                style='open-street-map'),
    margin={"r":0, "t":0, "l":0, "b":0},
                )
fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)


dir_plots = "./plots_html"
if callsign!="":
    fig.write_html(f"{dir_plots}/{date}_{callsign}_track_waypoint_fir.html")
else:
    if FIR_plot:
        if show_callsign:
            fig.write_html(f"{dir_plots}/{date}_trackARR_waypoint_fir_showAC.html")

        else: 
            if show_nearAirports:
                fig.write_html(f"{dir_plots}/{date}_trackARR_waypoint_fir_{hours[0]}_{hours[-1]}_nearAirports{dist_nearAirports}_std.html")
            else:
                fig.write_html(f"{dir_plots}/{date}_trackARR_waypoint_fir_{hours[0]}_{hours[-1]}.html")
    else:
        fig.write_html(f"{dir_plots}/{date}_trackARR_waypoint_mean.html")

fig.show()
