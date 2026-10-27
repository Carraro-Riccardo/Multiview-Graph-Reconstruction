"""Immagine di riferimento e centroidi fusi multi-vista VGGT.

Uso: python tools/debug/visualize_pointcloud_bboxes.py 7
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from graph_reconstruction.config import DEBUG_OUTPUT_DIR,IMAGES_DIR,WORK_DIR
from graph_reconstruction.pipeline.observed_graphs import build_all_observed_graphs
from graph_reconstruction.pipeline.vggt_3d_edges import (_candidate_files_for_image,
    _estimate_observation_geometry,_load_pointmap,build_vggt_search_roots)
from graph_reconstruction.shared.detection import discover_nodes

def sample(points,limit):
    points=points[np.isfinite(points).all(axis=1)]
    return points if len(points)<=limit else points[np.linspace(0,len(points)-1,limit,dtype=int)]

def equal_axes(ax,points):
    low,high=np.percentile(points,[1,99],axis=0); center=(low+high)/2; radius=max(float(np.max(high-low))/2,1e-6)
    ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[1]-radius,center[1]+radius)
    ax.set_zlim(center[2]-radius,center[2]+radius); ax.set_box_aspect((1,1,1))

def scene_map(node_id):
    path=WORK_DIR/"multiview"/f"node{node_id}"/"scene_nodes.json"
    if not path.exists(): return {},{}
    nodes=json.loads(path.read_text(encoding="utf-8")); obs_to_scene={}; codes={}
    for node in nodes:
        sid=node["scene_node_id"]; codes[sid]=node["fused_code"]
        for obs_id in node["supporting_observed_ids"]: obs_to_scene[obs_id]=sid
    return obs_to_scene,codes

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("node_id",help="7, 15_1, ...")
    p.add_argument("--reference",help="Vista mostrata a sinistra, es. node7_0003")
    p.add_argument("--max-points",type=int,default=70000,help="Punti grigi di contesto nel pannello 3D")
    p.add_argument("--elev",type=float,default=35); p.add_argument("--azim",type=float,default=-55)
    p.add_argument("--no-flip-x",action="store_true"); p.add_argument("--no-show",action="store_true"); args=p.parse_args()
    images=[Path(x) for x in discover_nodes(IMAGES_DIR).get(args.node_id,[])]
    if not images: raise SystemExit("Nessuna immagine trovata")
    reference=next((x for x in images if x.stem==Path(args.reference).stem),None) if args.reference else images[0]
    if reference is None: raise SystemExit("Immagine di riferimento non trovata")
    graphs=build_all_observed_graphs(images); graph_by_image={g.image_id:g for g in graphs}
    roots=build_vggt_search_roots(args.node_id,WORK_DIR/"multiview"/f"node{args.node_id}",WORK_DIR)
    loaded=[]
    for image in images:
        graph=graph_by_image.get(image.stem)
        if graph is None: continue
        for path in _candidate_files_for_image(image.stem,roots):
            pointmap=_load_pointmap(path)
            if pointmap is not None: loaded.append((image,graph,path,pointmap)); break
    if not loaded: raise SystemExit("Nessuna point map VGGT caricabile")
    obs_to_scene,codes=scene_map(args.node_id); scene_ids=sorted(set(obs_to_scene.values()))
    cmap=plt.colormaps["tab20"]; colors={sid:cmap(i/max(1,len(scene_ids)-1)) for i,sid in enumerate(scene_ids)}
    geometries=defaultdict(list); backgrounds=[]
    for image,graph,path,pointmap in loaded:
        backgrounds.append(sample(pointmap.reshape(-1,3),max(1,args.max_points//len(loaded))))
        for obs_id,obs in graph.nodes.items():
            geom=_estimate_observation_geometry(pointmap,list(obs.bbox))
            if geom: geometries[obs_to_scene.get(obs_id,obs_id)].append(geom)

    fig=plt.figure(figsize=(16,8)); ax_image=fig.add_subplot(121); ax3=fig.add_subplot(122,projection="3d")
    ref_graph=graph_by_image[reference.stem]; image_array=plt.imread(reference); ih,iw=image_array.shape[:2]
    ax_image.imshow(image_array); ax_image.axis("off"); ax_image.set_title(f"Immagine di riferimento - {reference.stem}")
    for obs_id,obs in ref_graph.nodes.items():
        sid=obs_to_scene.get(obs_id,obs_id); color=colors.get(sid,"red"); x1,y1,x2,y2=obs.bbox
        ax_image.add_patch(plt.Rectangle((x1*iw,y1*ih),(x2-x1)*iw,(y2-y1)*ih,fill=False,color=color,linewidth=2.2))
        ax_image.text(x1*iw,max(2,y1*ih-4),f"{sid}  code={codes.get(sid,obs.code)}",color="white",fontsize=8,
                      bbox=dict(facecolor=color,alpha=.85,edgecolor="none",pad=2))

    background=np.vstack(backgrounds); ax3.scatter(*background.T,s=.25,color="gray",alpha=.035,depthshade=False)
    for index,(sid,items) in enumerate(sorted(geometries.items())):
        color=colors.get(sid,cmap(index%20/19)); centers=np.asarray([g["centroid"] for g in items]); fused=np.median(centers,axis=0)
        ax3.scatter(*fused,s=145,color=color,edgecolor="black",linewidth=1.4,label=f"{sid} code={codes.get(sid,'?')}")
        ax3.text(*fused,f" {sid}",color=color,fontsize=9,fontweight="bold")
    equal_axes(ax3,background); ax3.view_init(elev=args.elev,azim=args.azim)
    if not args.no_flip_x: ax3.invert_xaxis()
    ax3.set(title=f"Centroidi fusi da {len(loaded)} viste",xlabel="X",ylabel="Y",zlabel="Z"); ax3.legend(fontsize=7,loc="upper left")
    fig.suptitle(f"VGGT node {args.node_id}: riferimento 2D e centroidi Re-ID fusi"); fig.tight_layout()
    destination=DEBUG_OUTPUT_DIR/f"node{args.node_id}"/"pointcloud_bboxes"; destination.mkdir(parents=True,exist_ok=True)
    output=destination/f"node{args.node_id}_fused_centroids.png"; fig.savefig(output,dpi=180,bbox_inches="tight")
    print(f"Viste fuse: {len(loaded)}/{len(images)}; centroidi fusi: {len(geometries)}\nSalvato: {output}")
    if args.no_show: plt.close(fig)
    else: plt.show()

if __name__=="__main__": main()
