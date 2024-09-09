import bpy

def find_loops_by_vert_co( o, co ):
    verts = o.data.vertices
    loopsInCo = []
    for poly in o.data.polygons:
        for i in poly.loop_indices:
            vCo = verts[ o.data.loops[ i ].vertex_index ].co[:]
            compare = [ round( e1, 3 ) == round( e2, 3 ) for e1, e2 in zip( vCo, co ) ]
            count   = len( [ e for e in compare if e ] ) == 3
            if count: loopsInCo.append( i )
            
    return loopsInCo

o = bpy.context.object

# Create a new vertex color layer
o.data.vertex_colors.new('Heat')
heat = o.data.vertex_colors['Heat']

coos2colors = [
    { 'co' : (1.0, 0.9999999403953552, -1.0),                  'color' : ( 0.5, 0.25, 0.25 ) },
    { 'co' : (1.0, -1.0, -1.0),                                'color' : ( 0.1, 0.6,  0.3  ) }, 
    { 'co' : (-1.0000001192092896, -0.9999998211860657, -1.0), 'color' : ( 0.9, 0.1,  0.2  ) },
    { 'co' : (-0.9999996423721313, 1.0000003576278687, -1.0),  'color' : ( 0.5, 0.8,  0.0  ) },
    { 'co' : (1.0000004768371582, 0.999999463558197, 1.0),     'color' : ( 0.4, 0.4,  0.9  ) },
    { 'co' : (0.9999993443489075, -1.0000005960464478, 1.0),   'color' : ( 0.1, 0.1,  0.6  ) },
    { 'co' : (-1.0000003576278687, -0.9999996423721313, 1.0),  'color' : ( 0.5, 0.5,  0.0  ) },
    { 'co' : (-0.9999999403953552, 1.0, 1.0),                  'color' : ( 0.5, 0.25, 0.25 ) }
]

for d in coos2colors:
    co, color = d['co'], d['color']
    
    vIndices = find_loops_by_vert_co( o, co )
    for vi in vIndices:
        heat.data[ vi ].color = color
