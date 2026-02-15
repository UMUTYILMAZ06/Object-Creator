using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(menuName = "Scene/PrefabLibrary")]
public class PrefabLibrary : ScriptableObject
{
    [System.Serializable]
    public class Entry
    {
        public string category;   
        public GameObject prefab; 
    }

    public List<Entry> entries = new List<Entry>();

    public GameObject GetPrefabForCategory(string category)
    {
        var entry = entries.Find(e => e.category == category);
        return entry != null ? entry.prefab : null;
    }
}