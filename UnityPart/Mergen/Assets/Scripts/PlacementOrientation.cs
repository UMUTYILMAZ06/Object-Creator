using UnityEngine;

public class PlacementOrientation : MonoBehaviour
{
    [Tooltip("The 'forward' direction in this object's local space. Default is Z+ (Vector3.forward).")]
    public Vector3 localForward = Vector3.forward;

    [Tooltip("The 'right' direction in this object's local space. Default is X+ (Vector3.right).")]
    public Vector3 localRight = Vector3.right;
}
