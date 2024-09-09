import folium
from folium import plugins
print (folium.__version__)
import ipywidgets
from ipywidgets import interact

# Helper functions

colors = [
    'black','green','orange','purple','darkred','darkgreen','darkpurple','darkblue',    'cadetblue',     'gray',
    'lightblue','lightgreen','red','pink','blue','red','black'
]

def utc_to_local(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    
def apply_timestamp(row):
    d = row.asDict()
    d["min_ts"] = utc_to_local(d["min_ts"])
    d["max_ts"] = utc_to_local(d["max_ts"])
    return Row(**d)


def apply_array_mapping(row):
    d = row.asDict()
    event_type_arr = row["event_type_arr"]
    d["mapped_event_type_arr"] = [mapping[i] for i in event_type_arr]
    return Row(**d)

def extract_trips(df_imsi, infra_conn_dev):
    if df_imsi.rdd.isEmpty():
        print ("No events with this IMSI")
        return
    lista = df_imsi.map(lambda x: (x[2], x[1], x[6], x[8])).collect()
    area_id_arr = lista[0][1]
    cell_id_arr = lista[0][2]
    mapped_event_type_arr = lista[0][2]
    timestamp_arr = lista[0][3]

    # Create ci_lac array
    ci_lac_list = zip(lista[0][0], lista[0][1])
    ci_lac_arr = [ci_lac[0] + "-" + ci_lac[1] for ci_lac in ci_lac_list]

    chain = zip(ci_lac_arr, timestamp_arr, mapped_event_type_arr)
    # chain to nametuple

    steps_list = [Section(x=get_centroid(infra_conn_dev, cilac)[0],y=get_centroid(infra_conn_dev, cilac)[1], cilac=cilac, timestamp=timestamp, \
                          record_type=record_type ) for cilac, timestamp, record_type in chain]
    print (steps_list)
    return steps_list


#####################################
# Get the data from parsed trips in csv
######################################

def parse_sms(path_to_csv, IMSI):
    parsed_sms_df = pd.read_csv(path_to_csv, sep=";", encoding='latin1')
    df_parsed_imsi = parsed_sms_df.loc[parsed_sms_df["IMSI"] == IMSI]

    trips_list = []
    # parsed trips format
    # [[(x, y), from_address, to_date, to_time], [(x, y), from_address, to_date, to_time]]

    for row in df_parsed_imsi.iterrows():
        index, data = row
        trips_list.append(data.tolist())
    return trips_list

def format_trips(trips_list):
    trips_formatted = []
    for trip in trips_list:
        from_address = trip[1]
        from_lon = float(trip[3].replace(",", "."))
        from_lat = float(trip[4].replace(",", "."))
        from_date = trip[5]
        from_time = trip[6]
        to_address = trip[7]
        to_lon = float(trip[8].replace(",", "."))
        to_lat = float(trip[9].replace(",", "."))
        to_date = trip[10]
        to_time = trip[11]

        new_trip = [[(from_lat, from_lon), from_address, from_date, from_time],
                    [(to_lat, to_lon), to_address, to_date, to_time]]

        trips_formatted.append(new_trip)
    return trips_formatted

# Second layer
# Extracted sms/calls in csv format
def plot_sms_trips(path_to_csv, IMSI, f):
    # Add to map
    trips_list = parse_sms(path_to_csv, int(IMSI))
    trips_formatted = format_trips(trips_list)

    radius = 100
    for trip in trips_formatted:
        for coords in trip:
            print (coords[0])
            popup_string = "address: {address}, date: {date}, time: {time}".format(address=coords[1], date=coords[2], time=coords[3])
            folium.Circle(coords[0], color="orange", radius=radius, popup = popup_string).add_to(f)
            radius +=30


            # First layer
# Choloropeth geojson for tiles

def add_tiles(f):
    f.choropleth(
        geo_data= os.environ['LAV_DIR'] +  "gis/mvg/mvg_tcs_part.geojson",
        name='tiles',
        fill_color='blue',
        line_color = "red",
        fill_opacity=0.1,
        line_opacity=0.4
    )
    
def add_centroids(f, is_centroids):
    if is_centroids:
        path_to_centroids = os.environ['LAV_DIR'] + "/gis/mvg/mvg_centroids_part.csv"
        list_centroids = add_centroids_to_map(path_to_centroids)
        for x, y, nr in list_centroids:
            folium.RegularPolygonMarker([y, x], popup=str(int(nr)), radius=2).add_to(f)
    
def add_centroids_to_map(path_to_centroids):
    df_centroids = pd.read_csv(path_to_centroids)
    df_part = df_centroids[["X", "Y", "NR"]]
    list_centroids = []
    for row in df_part.iterrows():
        index, data = row
        list_centroids.append(data.tolist())
    return list_centroids

# Third layer
# Events from the aggregator
def plot_all_coords(steps_imsi):
    print("length steps imsi",len(steps_imsi))
    coords_all = []
    cilacs = dict()
    for section in steps_imsi:
        for steps in section:
            for step in steps:
                if step.x:
                    try:
                        # in case record_type is in step
                        coords_all.append([step.cilac, step.timestamp, step.y, step.x, step.record_type])
                    except:
                        # in case no record_type is in step
                        coords_all.append([step.cilac, step.timestamp, step.y, step.x])
                    cilacs[step.cilac] = {"x": step.x, "y": step.y}
    return cilacs, coords_all


class ContinueLoop(Exception):
    pass

def add_aggregator_to_map(coords_all, infra_conn_dev, infra_conn_subway, intervals, f, m, other):
    #folium.PolyLine([[y, x] for cilac, ts, y, x in coords_all],color="gray", weight=2, popup="aggregator output").add_to(m)
    if len(coords_all[0]) == 4:
        for cilac, ts, y, x in coords_all:
            folium.RegularPolygonMarker([y, x], popup = cilac + " " + utc_to_local(ts), radius=30).add_to(m)
    elif len(coords_all[0]) == 5:
        radius = 100
        for cilac, ts, y, x, record in coords_all:
            try:
                if intervals:
                    for interval in intervals:
                        if ts >= interval[0] or ts <= interval[1]:
                            print ("contninuing")
                            raise ContinueLoop
            except ContinueLoop:
                continue
            #print cilac, record
            if cilac in underground_cilacs:
                folium.RegularPolygonMarker([y, x], color="green", popup = cilac + " " + utc_to_local(ts), radius=10).add_to(f)
                folium.RegularPolygonMarker([y, x], color="green", popup = cilac + " " + utc_to_local(ts), radius=10).add_to(other)
            elif cilac not in underground_cilacs:
                folium.RegularPolygonMarker([y, x], color="red", popup = cilac + " " + utc_to_local(ts), radius=10).add_to(f)
                folium.RegularPolygonMarker([y, x], color="red", popup = cilac + " " + utc_to_local(ts), radius=10).add_to(other)
            if record == "Online" or record == "SMS":
                add_bse_to_map(infra_conn_dev, infra_conn_subway, cilac, m)
                folium.Marker([y, x], popup = record + " " + cilac + " "+ utc_to_local(ts)).add_to(f)
                folium.Marker([y, x], popup = record + " " + cilac + " "+ utc_to_local(ts)).add_to(m)
                radius += 30
                



                
