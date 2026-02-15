using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(menuName = "Scene/MaterialLibrary")]
public class MaterialLibrary : ScriptableObject
{
    [System.Serializable]
    public class Entry
    {
        public string materialName; 
        public Material material;
    }

    public List<Entry> entries = new List<Entry>();

    public Material GetMaterial(string name)
    {
        var entry = entries.Find(e => e.materialName == name);
        return entry != null ? entry.material : null;
    }
}