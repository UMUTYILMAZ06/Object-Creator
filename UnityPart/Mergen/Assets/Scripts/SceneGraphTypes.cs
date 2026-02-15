using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class SceneGraphState
{
    public List<GraphNode> nodes = new();
    public List<GraphEdge> edges = new();
}

[Serializable]
public class GraphNode
{
    public string id;                
    public string key;               
    public string span;               
    public List<string> attributes = new();
    public TransformSnapshot transform = new();
}

[Serializable]
public class GraphEdge
{
    public string source;             
    public string target;             
    public string relation;           
    public string side;
    public float distance_m;
}

[Serializable]
public class TransformSnapshot
{
    public float[] pos = new float[3];
    public float[] rot = new float[3];
    public float[] scale = new float[3];
}