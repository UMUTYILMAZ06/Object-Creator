using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

[Serializable]
public class CommandMessage
{
    public string intent;               // "CreateObject", "PlaceObject", ...
    public CommandArgs args;            // category, reference_category, relation...

    public string model_full_id;        // metadata.csv'den seçilen fullId
    public string moved_span;           // RoleTagger: "glass", "pencil", "apple"
    public string reference_span;       // RoleTagger: "table", "desk", vs.
}

[Serializable]
public class CommandArgs
{
    public string category;
    public string reference_category;
    public string relation;
    public string side;

    public string moved_category;
    public string ref_category;
    public float distance;

    public int quantity = 1;
    public float distance_m = 0f;

    // Room settings
    public float room_width = 6f;
    public float room_length = 8f;
    public float room_height = 3f;

    // Rotate
    public float rotate_degrees = 0f;

    // Resize
    public float scale_factor = 1f;

    // Material
    public string material_name;
    public List<string> object_attribute = new();

    public string MovedCategory()
    => !string.IsNullOrEmpty(category) ? category : moved_category;

    public string RefCategory()
        => !string.IsNullOrEmpty(reference_category) ? reference_category : ref_category;

    public float DistanceMeters()
        => (distance_m != 0f) ? distance_m : distance;


}