using UnityEngine;

public class RoomController : MonoBehaviour
{
    [Header("Room Geometry (6 Cubes)")]
    public Transform floor;
    public Transform ceiling;
    public Transform wallFront;
    public Transform wallBack;
    public Transform wallLeft;
    public Transform wallRight;

    [Header("Default Size (meters)")]
    public float defaultWidth = 6f;   
    public float defaultLength = 8f; 
    public float defaultHeight = 3f; 

    [Header("Wall Thickness (meters)")]
    public float wallThickness = 0.2f;

    public float CurrentWidth  { get; private set; }
    public float CurrentLength { get; private set; }
    public float CurrentHeight { get; private set; }


    public Vector3 RoomCenter => transform.position;

    private void Start()
    {
        SetRoomSize(defaultWidth, defaultLength, defaultHeight);
    }

    public void SetRoomSize(float width, float length, float height)
    {
        if (width  <= 0f) width  = defaultWidth;
        if (length <= 0f) length = defaultLength;
        if (height <= 0f) height = defaultHeight;

        CurrentWidth  = width;
        CurrentLength = length;
        CurrentHeight = height;

        float halfW = width  * 0.5f;
        float halfL = length * 0.5f;
        float halfH = height * 0.5f;

   
        if (floor != null)
        {
        
            floor.localPosition = new Vector3(0f, 0f, 0f);
            floor.localRotation = Quaternion.identity;
            floor.localScale    = new Vector3(width, wallThickness, length);
        }

     
        if (ceiling != null)
        {
        
            ceiling.localPosition = new Vector3(0f, height, 0f);
            ceiling.localRotation = Quaternion.identity;
            ceiling.localScale    = new Vector3(width, wallThickness, length);
        }


        if (wallFront != null)
        {
            wallFront.localRotation = Quaternion.identity;
            wallFront.localPosition = new Vector3(0f, halfH, halfL);
            wallFront.localScale    = new Vector3(width, height, wallThickness);
        }

       
        if (wallBack != null)
        {
            wallBack.localRotation = Quaternion.identity;
            wallBack.localPosition = new Vector3(0f, halfH, -halfL);
            wallBack.localScale    = new Vector3(width, height, wallThickness);
 
        }

      
        if (wallLeft != null)
        {
            wallLeft.localRotation = Quaternion.identity;
            wallLeft.localPosition = new Vector3(-halfW, halfH, 0f);
            wallLeft.localScale    = new Vector3(wallThickness, height, length);
        }

      
        if (wallRight != null)
        {
            wallRight.localRotation = Quaternion.identity;
            wallRight.localPosition = new Vector3(halfW, halfH, 0f);
            wallRight.localScale    = new Vector3(wallThickness, height, length);
        }
    }

  
    public bool TryGetWallInfo(string side, out Vector3 center, out Vector3 inwardNormal)
    {
        Transform wall = null;

        switch (side)
        {
            case "front":
                wall = wallFront;
                break;
            case "back":
                wall = wallBack;
                break;
            case "left":
                wall = wallLeft;
                break;
            case "right":
                wall = wallRight;
                break;
            default:
                wall = wallBack;
                break;
        }

        if (wall == null)
        {
            center = transform.position;
            inwardNormal = transform.forward;
            return false;
        }

        center = wall.position;
  
        inwardNormal = wall.forward;
        return true;
    }

    public float GetWallThickness()
    {
        return wallThickness;
    }

    public float GetFloorTopY()
    {
        if (floor == null)
            return transform.position.y;

  
        var renderer = floor.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            return renderer.bounds.max.y;
        }

  
        float halfThicknessWorld = (floor.lossyScale.y * 0.5f);
        return floor.position.y + halfThicknessWorld;
    }
}
