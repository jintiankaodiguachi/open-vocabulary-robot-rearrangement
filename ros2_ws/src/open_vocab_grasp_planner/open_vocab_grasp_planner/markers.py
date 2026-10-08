from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def _p(x, y, z):
    p = Point(); p.x=float(x); p.y=float(y); p.z=float(z); return p


def delete_all(frame_id, stamp):
    m = Marker(); m.header.frame_id=frame_id; m.header.stamp=stamp; m.action=Marker.DELETEALL
    return MarkerArray(markers=[m])


def make_candidate_markers(frame_id, stamp, rows, selected_index=None):
    arr = MarkerArray()
    for i, row in enumerate(rows):
        cand = row["candidate"]
        feasible = bool(row.get("feasible", False))
        selected = selected_index is not None and i == selected_index
        gp = row["grasp_pose"].pose.position
        pp = row["pregrasp_pose"].pose.position

        s = Marker(); s.header.frame_id=frame_id; s.header.stamp=stamp
        s.ns="grasp_goal"; s.id=i; s.type=Marker.SPHERE; s.action=Marker.ADD
        s.pose.position=gp; s.pose.orientation.w=1.0; s.scale.x=s.scale.y=s.scale.z=0.025
        if selected: s.color.r,s.color.g,s.color.b,s.color.a=1.0,0.2,0.0,1.0
        elif feasible: s.color.r,s.color.g,s.color.b,s.color.a=0.1,0.9,0.1,0.9
        else: s.color.r,s.color.g,s.color.b,s.color.a=0.8,0.1,0.1,0.4
        arr.markers.append(s)

        a = Marker(); a.header.frame_id=frame_id; a.header.stamp=stamp
        a.ns="approach"; a.id=i; a.type=Marker.ARROW; a.action=Marker.ADD
        a.points=[_p(pp.x,pp.y,pp.z), _p(gp.x,gp.y,gp.z)]
        a.scale.x=0.008; a.scale.y=0.016; a.scale.z=0.020
        if selected: a.color.r,a.color.g,a.color.b,a.color.a=1.0,0.5,0.0,1.0
        elif feasible: a.color.r,a.color.g,a.color.b,a.color.a=0.1,0.8,1.0,0.9
        else: a.color.r,a.color.g,a.color.b,a.color.a=0.6,0.6,0.6,0.3
        arr.markers.append(a)

        t = Marker(); t.header.frame_id=frame_id; t.header.stamp=stamp
        t.ns="labels"; t.id=i; t.type=Marker.TEXT_VIEW_FACING; t.action=Marker.ADD
        t.pose.position.x=gp.x; t.pose.position.y=gp.y; t.pose.position.z=gp.z+0.05; t.pose.orientation.w=1.0
        t.scale.z=0.025; t.color.r=t.color.g=t.color.b=t.color.a=1.0
        t.text=f"#{cand.get('rank',i)} G={cand.get('model_score',0):.3f} W={cand.get('width_m',0)*1000:.0f}mm {'OK' if feasible else 'X'}"
        arr.markers.append(t)
    return arr
