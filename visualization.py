import random
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_all_with_pairs(cluster_num, clusters, signatures):
    original = []
    images = []
    for i in range(len(clusters[cluster_num])):
        orig = signatures[clusters[cluster_num][i]['orig']]['ptcoor']
        img = signatures[clusters[cluster_num][i]['image']]['ptcoor']
        original.append(orig)
        images.append(img)

    original= np.array(original)
    images = np.array(images)
    
    allcoors = [sig['ptcoor'] for sig in signatures]
    all = np.asarray(allcoors)
    
    x_edges = []
    y_edges = []
    z_edges = []

    for i in range(len(images)):
        x_coors = [images[i][0], original[i][0], None]
        x_edges += x_coors
        y_coors = [images[i][1], original[i][1], None]
        y_edges += y_coors
        z_coors = [images[i][2], original[i][2], None]
        z_edges += z_coors
        
    x = all
    Xax = x[:,0]
    Yax = x[:,1]
    Zax = x[:,2]
    x_lbl = ['gray'] * len(x)

    i = images
    Xix = i[:,0]
    Yix = i[:,1]
    Zix = i[:,2]
    i_lbl = ['red'] * len(i)

    o = original
    Xox = o[:,0]
    Yox = o[:,1]
    Zox = o[:,2]
    o_lbl = ['blue'] * len(o)

    lbl = np.append(i_lbl, o_lbl)
    x_tot = np.append(Xix, Xox)
    y_tot = np.append(Yix, Yox)
    z_tot = np.append(Zix, Zox)
    
    selected = go.Scatter3d(
        x=x_tot,
        y=y_tot,
        z=z_tot,
        mode='markers',
        marker=dict(
            size=4,
            color=lbl,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )

    allpoints = go.Scatter3d(
        x=Xax,
        y=Yax,
        z=Zax,
        mode='markers',
        marker=dict(
            size=4,
            color=x_lbl,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.1
        )
    )

    trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none'
    )

    data = [allpoints, selected, trace_edges]
    fig = go.Figure(data=data)

    fig.show()


def make_rot_trans(transf_space, signatures, thresh = None, axis = None):
    p_r = []
    p_t = []
    if thresh is None:
        for t in transf_space:
            ea = Rotation.from_matrix(t['R']).as_euler('xyz')
            ea_vector = list(ea)
            p_r.append(ea_vector)
            p_t.append(t['t'])
    else:
        for t in transf_space:
            if signatures[t['orig']]['ptcoor'][axis]>thresh:
                ea = Rotation.from_matrix(t['R']).as_euler('xyz')
                ea_vector = list(ea)
                p_r.append(ea_vector)
                p_t.append(t['t'])
    rots = np.array(p_r)
    trans = np.array(p_t)
    

    fig = go.Figure(data=[go.Scatter3d(
        x=rots[:,0],
        y=rots[:,1],
        z=rots[:,2],
        mode='markers',
        marker=dict(
            size=4
        )
    )])
    fig.update_layout(
        
        title=dict(text='Rotation space', font=dict(size=10), automargin=True, yref='paper'),
        scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )
    # tight layout
    fig.show()
    # tight layout
    
    
    fig = go.Figure(data=[go.Scatter3d(
        x=trans[:,0],
        y=trans[:,1],
        z=trans[:,2],
        mode='markers',
        marker=dict(
            size=4
        )
    )])

    fig.update_layout(
        
        title=dict(text='Translation space', font=dict(size=10), automargin=True, yref='paper'),
        scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )
    # tight layout
    fig.show()

            
def simple_plot(coordinates):
    Xax = coordinates[:,0]
    Yax = coordinates[:,1]
    Zax = coordinates[:,2]


    fig = go.Figure(data=[go.Scatter3d(
        x=Xax,
        y=Yax,
        z=Zax,
        mode='markers',
        marker=dict(
            size=4
        )
    )])

    # tight layout
    fig.show()
    

def vis_clusters(clusters, signatures, num_c):
    color_code = {'0': 'blue',
            '1': 'red',
            '2': 'lightblue',
            '3': 'lightgreen',
            '4': 'lightcyan'} 
    
    
    original = []
    color_map = []
    for j in range(num_c): #just check top 4 clusters
        for i in range(len(clusters[j])):
            orig = signatures[clusters[j][i]['orig']]['ptcoor']
            color = color_code[str(j)]
            original.append(orig)
            color_map.append(color)

    original= np.array(original)
    color_map = np.array(color_map)
    
    x = original

    Xax = x[:,0]
    Yax = x[:,1]
    Zax = x[:,2]

    fig = go.Figure(data=[go.Scatter3d(
        x=Xax,
        y=Yax,
        z=Zax,
        mode='markers',
        marker=dict(
            size=4,
            color=color_map              # set color to an array/list of desired values
        )
    )])

    # tight layout
    fig.show()
    
