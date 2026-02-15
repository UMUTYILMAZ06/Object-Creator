using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class SceneObject
{

    public string id;
    public string category;


    public Transform transform;
}

public class SceneMemory : MonoBehaviour
{

    public static SceneMemory Instance { get; private set; }


    [SerializeField]
    private List<SceneObject> _objects = new List<SceneObject>();


    private readonly Dictionary<string, int> _categoryCounters =
        new Dictionary<string, int>();

    public string Dump()
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("[SceneMemory Dump]");

        for (int i = 0; i < _objects.Count; i++)
        {
            var o = _objects[i];
            if (o == null || o.transform == null) continue;
            sb.AppendLine($"- id={o.id} category={o.category} name={o.transform.name}");
        }

        return sb.ToString();
    }

    private void Awake()
    {

        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("[SceneMemory] Multiple instances found, excess are being destroyed.");
            Destroy(gameObject);
            return;
        }

        Instance = this;
    }


    private string NormalizeKey(string category)
    {
        if (string.IsNullOrWhiteSpace(category))
            return "unknown";

        return category.Trim().ToLowerInvariant();
    }

    public string RegisterObject(string category, Transform t)
    {
        if (t == null)
        {
            Debug.LogWarning("[SceneMemory] RegisterObject: Transform null.");
            return null;
        }

        string key = NormalizeKey(category);

        if (!_categoryCounters.ContainsKey(key))
            _categoryCounters[key] = 0;

        _categoryCounters[key]++;
        string id = $"{key}_{_categoryCounters[key]}";

        var so = new SceneObject
        {
            id = id,
            category = key,
            transform = t
        };

        _objects.Add(so);

        Debug.Log($"[SceneMemory] Registered {id} (category={key}).");
        return id;
    }


    public void Unregister(Transform t)
    {
        if (t == null)
            return;

        int removed = _objects.RemoveAll(o => o.transform == t);
        if (removed > 0)
        {
            Debug.Log($"[SceneMemory] Unregistered {removed} object(s) for transform {t.name}.");
        }
    }


    public SceneObject GetFirstByCategory(string category)
    {
        if (string.IsNullOrEmpty(category))
            return null;

        string key = NormalizeKey(category);

        return _objects.Find(o => o.category == key && o.transform != null);
    }


    public List<SceneObject> GetAllByCategory(string category)
    {
        var result = new List<SceneObject>();

        if (string.IsNullOrEmpty(category))
            return result;

        string key = NormalizeKey(category);

        foreach (var o in _objects)
        {
            if (o.category == key && o.transform != null)
                result.Add(o);
        }

        return result;
    }


    public SceneObject GetById(string id)
    {
        if (string.IsNullOrEmpty(id))
            return null;

        return _objects.Find(o => o.id == id && o.transform != null);
    }


    [ContextMenu("Log All Objects")]
    public void LogAllObjects()
    {
        if (_objects.Count == 0)
        {
            Debug.Log("[SceneMemory] [Info] No saved objects found.");
            return;
        }

        Debug.Log("[SceneMemory] Saved objects:");
        foreach (var o in _objects)
        {
            if (o.transform == null)
                continue;

            Debug.Log($" - {o.id} (category={o.category}, pos={o.transform.position})");
        }
    }
    public SceneObject FindBestMatchByNameOrSpan(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return null;

        string q = NormalizeKey(query);


        var byId = GetById(q);
        if (byId != null) return byId;

 
        var byCat = GetFirstByCategory(q);
        if (byCat != null) return byCat;


        SceneObject best = null;

 
        for (int i = _objects.Count - 1; i >= 0; i--)
        {
            var o = _objects[i];
            if (o == null || o.transform == null) continue;

            string tname = NormalizeKey(o.transform.name);

            if (tname.Contains(q))
            {
                best = o;
                break;
            }
        }

        return best;
    }

    public SceneObject GetMostRecentByCategory(string category)
    {
        if (string.IsNullOrWhiteSpace(category))
            return null;

        string key = NormalizeKey(category);

 
        for (int i = _objects.Count - 1; i >= 0; i--)
        {
            var o = _objects[i];
            if (o != null && o.transform != null && o.category == key)
                return o;
        }

        return null;
    }
    
}
