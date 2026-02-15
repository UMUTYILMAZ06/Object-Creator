using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class SceneGraphManager : MonoBehaviour
{
    public string fileName = "scene_graph.json";
    public string roomNodeId = "room_1";

    public SceneGraphState state = new SceneGraphState();

    private string FilePath => Path.Combine(Application.persistentDataPath, fileName);

    private void Awake()
    {
        Load();
        EnsureRoomNode();
        Save();
    }
    public void UpsertRoomEdge(string objectId)
    {
        if (string.IsNullOrEmpty(objectId)) return;
        EnsureRoomNode();
        UpsertEdge(objectId, roomNodeId, "in_room", "none", 0f);
    }

    public void UpsertNodeFromTransform(string id, string key, string span, List<string> attrs, Transform t)
    {
        if (string.IsNullOrEmpty(id) || t == null) return;

        var n = FindNode(id);
        if (n == null)
        {
            n = new GraphNode { id = id };
            state.nodes.Add(n);
        }

        n.key = key ?? "";
        n.span = span ?? "";

        n.attributes = attrs != null ? new List<string>(attrs) : new List<string>();

        WriteSnapshot(n.transform, t);
        Save();
    }

    public void UpdateNodeTransform(string id, Transform t)
    {
        if (string.IsNullOrEmpty(id) || t == null) return;
        var n = FindNode(id);
        if (n == null) return;

        WriteSnapshot(n.transform, t);
        Save();
    }

    public void SetRelationAndParent(string sourceId, Transform sourceT,
                                     string targetId, Transform targetT,
                                     string relation, string side, float dist)
    {
        if (string.IsNullOrEmpty(sourceId) || string.IsNullOrEmpty(targetId)) return;

 
        UpsertEdge(sourceId, targetId, relation, side, dist);


        if (relation == "on_top_of" && sourceT != null && targetT != null)
        {
            sourceT.SetParent(targetT, true); 
        }
        else if (sourceT != null)
        {

            if (sourceT.parent != null && sourceT.parent == targetT)
                sourceT.SetParent(null, true);
        }

        Save();
    }

    public void HandleDeleteWithStacking(string deletedId, RoomController roomController)
    {
        if (string.IsNullOrEmpty(deletedId)) return;

    
        var children = GetChildrenOnTopOf(deletedId);

        foreach (var childId in children)
        {
          
            RemoveEdges(childId, deletedId, "on_top_of");

 
            var childGO = GameObject.Find(childId);
            if (childGO != null)
            {
                childGO.transform.SetParent(null, true);
                SnapToFloor(childGO.transform, roomController);
                UpdateNodeTransform(childId, childGO.transform);
            }
        }


        RemoveAllEdgesWith(deletedId);
        RemoveNode(deletedId);

        Save();
    }

    // -------------------- Internals --------------------

    private void EnsureRoomNode()
    {
        var room = FindNode(roomNodeId);
        if (room == null)
        {
            room = new GraphNode
            {
                id = roomNodeId,
                key = "room",
                span = "room",
                attributes = new List<string>()
            };
            room.transform.pos[0] = 0; room.transform.pos[1] = 0; room.transform.pos[2] = 0;
            room.transform.rot[0] = 0; room.transform.rot[1] = 0; room.transform.rot[2] = 0;
            room.transform.scale[0] = 1; room.transform.scale[1] = 1; room.transform.scale[2] = 1;
            state.nodes.Add(room);
        }
    }

    private GraphNode FindNode(string id)
        => state.nodes.Find(n => n.id == id);

    private void UpsertEdge(string source, string target, string relation, string side, float dist)
    {
        var e = state.edges.Find(x => x.source == source && x.target == target && x.relation == relation);
        if (e == null)
        {
            e = new GraphEdge { source = source, target = target, relation = relation };
            state.edges.Add(e);
        }

        e.side = side ?? "none";
        e.distance_m = dist;
    }

    private List<string> GetChildrenOnTopOf(string baseId)
    {
        var outList = new List<string>();
        foreach (var e in state.edges)
        {
            if (e.relation == "on_top_of" && e.target == baseId)
                outList.Add(e.source);
        }
        return outList;
    }

    private void RemoveEdges(string source, string target, string relation)
    {
        state.edges.RemoveAll(e => e.source == source && e.target == target && e.relation == relation);
    }

    private void RemoveAllEdgesWith(string id)
    {
        state.edges.RemoveAll(e => e.source == id || e.target == id);
    }

    private void RemoveNode(string id)
    {
        state.nodes.RemoveAll(n => n.id == id);
    }

    private void WriteSnapshot(TransformSnapshot snap, Transform t)
    {
        var p = t.position;
        var r = t.eulerAngles;
        var s = t.localScale;

        snap.pos[0] = p.x; snap.pos[1] = p.y; snap.pos[2] = p.z;
        snap.rot[0] = r.x; snap.rot[1] = r.y; snap.rot[2] = r.z;
        snap.scale[0] = s.x; snap.scale[1] = s.y; snap.scale[2] = s.z;
    }

    private void SnapToFloor(Transform target, RoomController roomController)
    {
        if (target == null || roomController == null) return;

        // Renderer bounds bul
        var renderers = target.GetComponentsInChildren<Renderer>();
        if (renderers == null || renderers.Length == 0) return;

        Bounds b = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
            b.Encapsulate(renderers[i].bounds);

        float floorTopY = roomController.GetFloorTopY();
        float bottomY = b.min.y;
        float offsetY = floorTopY - bottomY;

        target.position += new Vector3(0f, offsetY, 0f);
    }

    // -------------------- Persistence --------------------

    public void Save()
    {
        try
        {
            var json = JsonUtility.ToJson(state, true);
            File.WriteAllText(FilePath, json);
        }
        catch (Exception e)
        {
            Debug.LogWarning("[SceneGraphManager] Save failed: " + e.Message);
        }
    }

    public void Load()
    {
        try
        {
            if (!File.Exists(FilePath))
            {
                state = new SceneGraphState();
                return;
            }

            var json = File.ReadAllText(FilePath);
            state = JsonUtility.FromJson<SceneGraphState>(json) ?? new SceneGraphState();
        }
        catch (Exception e)
        {
            Debug.LogWarning("[SceneGraphManager] Load failed: " + e.Message);
            state = new SceneGraphState();
        }
    }
}
