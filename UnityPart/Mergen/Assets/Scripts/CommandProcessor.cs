using System.Collections.Generic;
using System.IO;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class CommandProcessor : MonoBehaviour
{
    [Header("References")]
    public FileCommandReceiver receiver;
    public SceneMemory sceneMemory;
    public PrefabLibrary prefabLibrary;
    public RoomController roomController;
    public MaterialLibrary materialLibrary;


    [Header("Scene Graph")]
    public SceneGraphManager sceneGraph;

    [Header("Defaults")]
    [Tooltip("Reference point where objects without explicit position info will be spawned.")]
    public Transform defaultSpawnPoint;

    [Header("ShapeNet Source (disk)")]
    [Tooltip("Folder containing ShapeNet .obj files (full path). Example: D:\\Shapenet\\ShapeNetSem-backup\\models-OBJ\\models")]
    public string shapeNetObjFolder = @"D:\Shapenet\ShapeNetSem-backup\models-OBJ\models";

    [Header("ShapeNet Import (Unity Project)")]
    [Tooltip("Folder inside the Unity project where imports will be placed (must start with Assets). Example: Assets/ImportedShapeNet")]
    public string shapeNetImportFolder = "Assets/ImportedShapeNet";

    [Tooltip("Default Euler rotation (degrees) applied when placing ShapeNet objects into the scene. Example: X = -90.")]
    public Vector3 shapeNetDefaultRotationEuler = new Vector3(-90f, 0f, 0f);

    [Tooltip("Global uniform scale multiplier applied to all ShapeNet objects.")]
    public float shapeNetUniformScale = 0.5f;

    [Header("ShapeNet Normalization")]
    [Tooltip("Automatically normalize based on bounds (largest dimension = targetMaxSize).")]
    public bool autoNormalizeShapeNetScale = true;

    [Tooltip("When auto-normalize is enabled, the model's largest dimension will be scaled to this size.")]
    public float targetMaxSize = 2.0f;

    [Tooltip("Recenter the pivot (transform.position) to the model's geometric center.")]
    public bool recenterShapeNetPivot = true;


    private const float DEFAULT_GAP = 1.5f;

    private void Awake()
    {
        if (receiver != null)
            receiver.OnCommandReceived += HandleCommand;
    }

    private void OnDestroy()
    {
        if (receiver != null)
            receiver.OnCommandReceived -= HandleCommand;
    }

    private void HandleCommand(CommandMessage cmd)
    {
        if (cmd == null || cmd.args == null)
        {
            Debug.LogWarning("[CommandProcessor] Empty command.");
            return;
        }

        Debug.Log($"[CommandProcessor] Handling intent: {cmd.intent}");

        switch (cmd.intent)
        {
            case "CreateObject":
                HandleCreateObject(cmd);
                break;

            case "PlaceObject":
                HandlePlaceObject(cmd);
                break;

            case "RotateObject":
                HandleRotateObject(cmd);
                break;

            case "ResizeObject":
                HandleResizeObject(cmd);
                break;

            case "DeleteObject":
                HandleDeleteObject(cmd);
                break;

            case "SetRoom":
                HandleSetRoom(cmd.args);
                break;

            case "SetMaterial":
                HandleSetMaterial(cmd);
                break;

            default:
                Debug.LogWarning($"[CommandProcessor] Unknown intent: {cmd.intent}");
                break;
        }
    }

    private string GetIdFromTransform(Transform t)
    {
        if (t == null) return null;
        return t.gameObject != null ? t.gameObject.name : null;
    }
    private void UpdateGraphSubtree(Transform root)
    {
        if (sceneGraph == null || root == null) return;

        var stack = new Stack<Transform>();
        stack.Push(root);

        while (stack.Count > 0)
        {
            var cur = stack.Pop();
            var id = GetIdFromTransform(cur);
            if (!string.IsNullOrEmpty(id))
                sceneGraph.UpdateNodeTransform(id, cur);

            foreach (Transform ch in cur)
                stack.Push(ch);
        }
    }


    private Vector3 GetDefaultSpawnPosition()
    {
        if (defaultSpawnPoint != null)
            return defaultSpawnPoint.position;

        if (roomController != null)
        {
            float floorY = roomController.GetFloorTopY();
            return new Vector3(0f, floorY, 0f);
        }

        return Vector3.zero;
    }


    private Transform SpawnShapeNetModelFromFullId(CommandMessage cmd, Vector3 position)
    {
        if (cmd == null || string.IsNullOrEmpty(cmd.model_full_id))
            return null;

        if (string.IsNullOrEmpty(shapeNetObjFolder) || !Directory.Exists(shapeNetObjFolder))
        {
            Debug.LogWarning("[CommandProcessor] ShapeNet obj folder not found: " + shapeNetObjFolder);
            return null;
        }

        string fullId = cmd.model_full_id;          
        string idPart = fullId;

        if (idPart.StartsWith("wss."))
            idPart = idPart.Substring(4);

        if (idPart.Contains("/"))
            idPart = idPart.Split('/')[0];

        string objFileName = idPart + ".obj";
        string srcObjPath = Path.Combine(shapeNetObjFolder, objFileName);

        if (!File.Exists(srcObjPath))
        {
            Debug.LogWarning($"[CommandProcessor] Source OBJ not found for fullId={fullId}, path={srcObjPath}");
            return null;
        }

#if UNITY_EDITOR

        if (string.IsNullOrEmpty(shapeNetImportFolder))
            shapeNetImportFolder = "Assets/ImportedShapeNet";

        if (!shapeNetImportFolder.StartsWith("Assets"))
            shapeNetImportFolder = "Assets/" + shapeNetImportFolder.TrimStart('/', '\\');

        string assetsRoot = Application.dataPath; 
        string relativeInsideAssets = shapeNetImportFolder.Substring("Assets".Length)
            .TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        string importFolderAbs = string.IsNullOrEmpty(relativeInsideAssets)
            ? assetsRoot
            : Path.Combine(assetsRoot, relativeInsideAssets);

        Directory.CreateDirectory(importFolderAbs);

        string dstObjPathAbs = Path.Combine(importFolderAbs, objFileName);
        File.Copy(srcObjPath, dstObjPathAbs, true);

        string srcMtlPath = Path.Combine(shapeNetObjFolder, idPart + ".mtl");
        if (File.Exists(srcMtlPath))
        {
            string dstMtlPathAbs = Path.Combine(importFolderAbs, idPart + ".mtl");
            File.Copy(srcMtlPath, dstMtlPathAbs, true);
        }


        string assetObjPath = Path.Combine(shapeNetImportFolder, objFileName)
            .Replace("\\", "/");


        AssetDatabase.ImportAsset(assetObjPath, ImportAssetOptions.ForceUpdate);
        GameObject asset = AssetDatabase.LoadAssetAtPath<GameObject>(assetObjPath);

        if (asset == null)
        {
            Debug.LogWarning("[CommandProcessor] Could not load GameObject from asset path: " + assetObjPath);
            return null;
        }

        GameObject go = Instantiate(asset);
        go.transform.position = position;
        go.transform.localEulerAngles = shapeNetDefaultRotationEuler;

        if (shapeNetUniformScale > 0f && Mathf.Abs(shapeNetUniformScale - 1f) > 0.0001f)
        {
            go.transform.localScale *= shapeNetUniformScale;
        }

        AutoNormalizeAndCenterShapeNet(go.transform);

        var parent = GetOrCreateImportedRoot();
        go.transform.SetParent(parent, true);

        Debug.Log($"[CommandProcessor] Spawned ShapeNet model from asset: {assetObjPath}");
        return go.transform;
#else
        Debug.LogWarning("[CommandProcessor] ShapeNet runtime import only works inside Unity Editor.");
        return null;
#endif
    }

    private Transform GetOrCreateImportedRoot()
    {
        const string rootName = "ImportedRuntimeModels";
        var existing = GameObject.Find(rootName);
        if (existing != null)
            return existing.transform;

        var root = new GameObject(rootName);
        return root.transform;
    }

    private void AutoNormalizeAndCenterShapeNet(Transform root)
    {
        if (root == null)
            return;

        if (!TryGetWorldBounds(root, out Bounds b))
            return;

        if (autoNormalizeShapeNetScale)
        {
            float maxSize = Mathf.Max(b.size.x, b.size.y, b.size.z);
            if (maxSize > 0.0001f && targetMaxSize > 0f)
            {
                float factor = targetMaxSize / maxSize;
                root.localScale *= factor;
                TryGetWorldBounds(root, out b);
            }
        }

        if (recenterShapeNetPivot)
        {
            Vector3 worldCenter = b.center;
            Vector3 offset = worldCenter - root.position;

            foreach (Transform child in root)
                child.position -= offset;
        }
    }


    private void HandleCreateObject(CommandMessage cmd)
    {
        var a = cmd.args;

        if (prefabLibrary == null && string.IsNullOrEmpty(shapeNetObjFolder))
        {
            Debug.LogWarning("[CommandProcessor] No PrefabLibrary and no ShapeNet folder assigned.");
            return;
        }

        int count = (a.quantity <= 0) ? 1 : a.quantity;

        for (int i = 0; i < count; i++)
        {
            Vector3 spawnPos = GetDefaultSpawnPosition();
            Transform objTransform = null;
            GameObject objGO = null;

  
            objTransform = SpawnShapeNetModelFromFullId(cmd, spawnPos);
            if (objTransform != null)
            {
                objGO = objTransform.gameObject;
            }
            else
            {
               
                if (prefabLibrary == null)
                {
                    Debug.LogWarning("[CommandProcessor] No PrefabLibrary assigned (ShapeNet also failed).");
                    return;
                }

                string spawnKey = !string.IsNullOrEmpty(cmd.moved_span)
                    ? cmd.moved_span
                    : a.category;

                var prefab = prefabLibrary.GetPrefabForCategory(spawnKey);

                if (prefab == null && !string.IsNullOrEmpty(a.category) && a.category != spawnKey)
                    prefab = prefabLibrary.GetPrefabForCategory(a.category);

                if (prefab == null)
                {
                    Debug.LogWarning($"[CommandProcessor] No prefab for key (span/category): {spawnKey} / {a.category}");
                    return;
                }

                objGO = Instantiate(prefab);
                objTransform = objGO.transform;
                objTransform.position = spawnPos;
            }
            string id = null;
            string memoryKey = !string.IsNullOrEmpty(cmd.moved_span) ? cmd.moved_span : a.category;
           
            if (sceneMemory != null && objTransform != null)
            {
                id = sceneMemory.RegisterObject(memoryKey, objTransform);
                objGO.name = id;
            }
           
            if (sceneGraph != null && objTransform != null && !string.IsNullOrEmpty(id))
            {
                var attrs = (a.object_attribute != null) ? a.object_attribute : new List<string>();
                sceneGraph.UpsertNodeFromTransform(
                    id: id,
                    key: memoryKey,
                    span: cmd.moved_span,
                    attrs: attrs,
                    t: objTransform
                );
                sceneGraph.UpsertRoomEdge(id);
            }
            bool placedByRelation = false;

            if (!string.IsNullOrEmpty(a.relation))
            {
                if (a.relation == "near_wall")
                {
                    PlaceNearWallForTransform(objTransform, a);
                    placedByRelation = true;
                }
                else if (!string.IsNullOrEmpty(a.reference_category) ||
                         !string.IsNullOrEmpty(cmd.reference_span))
                {
                    HandlePlaceObjectForTransform(objTransform, cmd);
                    placedByRelation = true;
                }
            }

            if (!placedByRelation)
                SnapToFloor(objTransform);
        }
    }



    private void HandlePlaceObject(CommandMessage cmd)
    {
        var a = cmd.args;

        if (sceneMemory == null)
        {
            Debug.LogWarning("[CommandProcessor] No SceneMemory assigned.");
            return;
        }

        var target = (!string.IsNullOrEmpty(cmd.moved_span))
            ? sceneMemory.GetFirstByCategory(cmd.moved_span)
            : null;

        if (target == null && !string.IsNullOrEmpty(a.category))
            target = sceneMemory.GetFirstByCategory(a.category);

        if (target == null)
        {
            Debug.LogWarning(
                $"[CommandProcessor] PlaceObject: no target for moved_span={cmd.moved_span} / category={a.category}");
            return;
        }

        if (a.relation == "near_wall")
        {
            Vector3 pos = ComputeNearWallPosition(a, target.transform);
            target.transform.position = pos;
            UpdateGraphSubtree(target.transform);
            return;
        }

        var reference = (!string.IsNullOrEmpty(cmd.reference_span))
            ? sceneMemory.GetFirstByCategory(cmd.reference_span)
            : null;

        if (reference == null && !string.IsNullOrEmpty(a.reference_category))
            reference = sceneMemory.GetFirstByCategory(a.reference_category);

        if (reference == null)
        {
            Debug.LogWarning(
                $"[CommandProcessor] PlaceObject: reference missing. " +
                $"ref_span={cmd.reference_span}, reference_category={a.reference_category}");
            return;
        }

        Vector3 newPos = ComputePlacementPosition(a, target.transform, reference.transform);
        target.transform.position = newPos;
        if (sceneGraph != null)
        {
            string srcId = GetIdFromTransform(target.transform);
            string tgtId = GetIdFromTransform(reference.transform);

            sceneGraph.SetRelationAndParent(
                sourceId: srcId,
                sourceT: target.transform,
                targetId: tgtId,
                targetT: reference.transform,
                relation: a.relation,
                side: a.side,
                dist: a.distance_m
            );
        }
        else
        {
            
            if (a.relation == "on_top_of")
                target.transform.SetParent(reference.transform, true);
        }

       
        UpdateGraphSubtree(reference.transform);
        UpdateGraphSubtree(target.transform);
    }




    private void HandlePlaceObjectForTransform(Transform targetTransform, CommandMessage cmd)
    {
        var a = cmd.args;

        if (sceneMemory == null)
            return;

        var reference = (!string.IsNullOrEmpty(cmd.reference_span))
            ? sceneMemory.GetFirstByCategory(cmd.reference_span)
            : null;

        if (reference == null && !string.IsNullOrEmpty(a.reference_category))
            reference = sceneMemory.GetFirstByCategory(a.reference_category);

        if (reference == null)
        {
            Debug.LogWarning(
                $"[CommandProcessor] PlaceObjectForTransform: reference missing. " +
                $"ref_span={cmd.reference_span}, reference_category={a.reference_category}");
            return;
        }

        Vector3 newPos = ComputePlacementPosition(a, targetTransform, reference.transform);
        targetTransform.position = newPos;
        if (sceneGraph != null)
        {
            string srcId = GetIdFromTransform(targetTransform);
            string tgtId = GetIdFromTransform(reference.transform);

            sceneGraph.SetRelationAndParent(
                sourceId: srcId,
                sourceT: targetTransform,
                targetId: tgtId,
                targetT: reference.transform,
                relation: a.relation,
                side: a.side,
                dist: a.distance_m
            );
        }
        else
        {
            if (a.relation == "on_top_of")
                targetTransform.SetParent(reference.transform, true);
        }

        UpdateGraphSubtree(reference.transform);
        UpdateGraphSubtree(targetTransform);
    }

    private void PlaceNearWallForTransform(Transform targetTransform, CommandArgs a)
    {
        Vector3 pos = ComputeNearWallPosition(a, targetTransform);
        targetTransform.position = pos;
    }



    private void HandleRotateObject(CommandMessage cmd)
    {
        var a = cmd.args;

        if (sceneMemory == null)
        {
            Debug.LogWarning("[CommandProcessor] No SceneMemory assigned.");
            return;
        }

        var target = (!string.IsNullOrEmpty(cmd.moved_span))
            ? sceneMemory.GetFirstByCategory(cmd.moved_span)
            : null;

        if (target == null && !string.IsNullOrEmpty(a.category))
            target = sceneMemory.GetFirstByCategory(a.category);

        if (target == null)
        {
            Debug.LogWarning(
                $"[CommandProcessor] RotateObject: no target for moved_span={cmd.moved_span} / category={a.category}");
            return;
        }

        if (Mathf.Abs(a.rotate_degrees) > 0.01f)
        {
            float angle = a.rotate_degrees;

            if (!string.IsNullOrEmpty(a.side))
            {
                if (a.side == "left")
                    angle = -Mathf.Abs(angle);
                else if (a.side == "right")
                    angle = Mathf.Abs(angle);
            }

            target.transform.Rotate(0f, angle, 0f, Space.World);
            UpdateGraphSubtree(target.transform);
            return;
        }

        if (!string.IsNullOrEmpty(a.reference_category) || !string.IsNullOrEmpty(cmd.reference_span))
        {
            var reference = (!string.IsNullOrEmpty(cmd.reference_span))
                ? sceneMemory.GetFirstByCategory(cmd.reference_span)
                : null;

            if (reference == null && !string.IsNullOrEmpty(a.reference_category))
                reference = sceneMemory.GetFirstByCategory(a.reference_category);

            if (reference == null)
            {
                Debug.LogWarning(
                    $"[CommandProcessor] RotateObject: reference missing for " +
                    $"ref_span={cmd.reference_span} / reference_category={a.reference_category}");
                return;
            }

            Vector3 targetPos = target.transform.position;
            Vector3 refPos = reference.transform.position;

            Vector3 dir = (refPos - targetPos);
            dir.y = 0f;
            if (dir.sqrMagnitude < 0.0001f)
                return;

            target.transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
        }
    }

  

    private void HandleResizeObject(CommandMessage cmd)
    {
        var a = cmd.args;

        if (sceneMemory == null)
        {
            Debug.LogWarning("[CommandProcessor] No SceneMemory assigned.");
            return;
        }

        var target = (!string.IsNullOrEmpty(cmd.moved_span))
            ? sceneMemory.GetFirstByCategory(cmd.moved_span)
            : null;

        if (target == null && !string.IsNullOrEmpty(a.category))
            target = sceneMemory.GetFirstByCategory(a.category);

        if (target == null)
        {
            Debug.LogWarning(
                $"[CommandProcessor] ResizeObject: no target for moved_span={cmd.moved_span} / category={a.category}");
            return;
        }

        float factor = (a.scale_factor <= 0f) ? 1f : a.scale_factor;
        target.transform.localScale *= factor;
        UpdateGraphSubtree(target.transform);
    }



    private void HandleDeleteObject(CommandMessage cmd)
    {
        var a = cmd.args;
        if (sceneMemory == null) return;

        List<SceneObject> all = null;

        if (!string.IsNullOrEmpty(cmd.moved_span))
            all = sceneMemory.GetAllByCategory(cmd.moved_span);

        if ((all == null || all.Count == 0) && !string.IsNullOrEmpty(a.category))
            all = sceneMemory.GetAllByCategory(a.category);

        if (all == null || all.Count == 0) return;

        int q = a.quantity;

        if (q <= 0)
        {
            foreach (var so in all)
                DeleteOneWithStackRules(so);
        }
        else
        {
            int toDelete = Mathf.Min(q, all.Count);
            for (int i = 0; i < toDelete; i++)
                DeleteOneWithStackRules(all[i]);
        }
    }

    private void DeleteOneWithStackRules(SceneObject so)
    {
        if (so == null || so.transform == null)
            return;

        var baseObj = so.transform;
        string deletedId = GetIdFromTransform(baseObj);


        if (sceneGraph != null && !string.IsNullOrEmpty(deletedId))
        {
            sceneGraph.HandleDeleteWithStacking(deletedId, roomController);

        }
        else
        {

            var children = new List<Transform>();
            foreach (Transform ch in baseObj)
                children.Add(ch);

            foreach (var ch in children)
            {
                ch.SetParent(null, true);
                SnapToFloor(ch);
            }
        }


        sceneMemory.Unregister(baseObj);
        Destroy(baseObj.gameObject);
    }

 

    private void HandleSetRoom(CommandArgs a)
    {
        if (roomController == null)
        {
            Debug.LogWarning("[CommandProcessor] SetRoom: no RoomController assigned.");
            return;
        }

        float w = (a.room_width > 0f) ? a.room_width : roomController.defaultWidth;
        float l = (a.room_length > 0f) ? a.room_length : roomController.defaultLength;
        float h = (a.room_height > 0f) ? a.room_height : roomController.defaultHeight;

        roomController.SetRoomSize(w, l, h);
    }


    private void HandleSetMaterial(CommandMessage cmd)
    {
        var a = cmd.args;

        if (sceneMemory == null || materialLibrary == null)
        {
            Debug.LogWarning("[CommandProcessor] SetMaterial: missing SceneMemory or MaterialLibrary.");
            return;
        }


        string span = !string.IsNullOrEmpty(cmd.moved_span) ? cmd.moved_span : cmd.reference_span;

        var target = ObjectResolver.ResolveTarget(sceneMemory, span, a.category, null);

        if (target == null)
        {
            Debug.LogWarning(
                $"[CommandProcessor] SetMaterial: no target for moved_span={cmd.moved_span} / reference_span={cmd.reference_span} / category={a.category}");
            return;
        }
        var targetGo = target.transform.gameObject;


        var mat = materialLibrary.GetMaterial(a.material_name);
        if (mat == null)
        {
            Debug.LogWarning($"[CommandProcessor] SetMaterial: no material for name={a.material_name}");
            return;
        }

        var renderers = target.transform.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            Debug.LogWarning("[CommandProcessor] SetMaterial: target has no Renderer.");
            return;
        }

        foreach (var r in renderers)
            r.material = mat;

        UpdateGraphSubtree(target.transform);
    }

 

    private Vector3 ComputePlacementPosition(CommandArgs a, Transform target, Transform reference)
    {

        float gap;
        if (a.relation == "on_top_of" || a.relation == "under")
        {

            gap = (a.distance_m > 0f) ? a.distance_m : 0f;
        }
        else
        {
 
            gap = (a.distance_m <= 0f) ? DEFAULT_GAP : a.distance_m;
        }

        Vector3 dir = GetDirectionForRelation(a.relation, a.side, reference);


        if (a.relation != "on_top_of" && a.relation != "under")
            dir.y = 0f;

        if (dir == Vector3.zero)
            dir = reference.forward;

        dir.Normalize();

        bool hasRefBounds = TryGetWorldBounds(reference, out Bounds refBounds);
        bool hasTgtBounds = TryGetWorldBounds(target, out Bounds tgtBounds);

        if (!hasRefBounds || !hasTgtBounds)
        {
            Vector3 fallbackPos = reference.position + dir * gap;
            if (a.relation != "on_top_of" && a.relation != "under")
                fallbackPos.y = reference.position.y;
            return fallbackPos;
        }

        float refHalf = GetHalfSizeAlong(refBounds, dir);
        float tgtHalf = GetHalfSizeAlong(tgtBounds, -dir);

        float centerDistance = refHalf + gap + tgtHalf;
        Vector3 pos = reference.position + dir * centerDistance;

  
        if (a.relation != "on_top_of" && a.relation != "under")
            pos.y = reference.position.y;

        return pos;
    }

    private Vector3 ComputeNearWallPosition(CommandArgs a, Transform target)
    {
        if (roomController == null)
        {
            Debug.LogWarning("[CommandProcessor] near_wall: no RoomController.");
            return target.position;
        }

        if (!roomController.TryGetWallInfo(a.side, out Vector3 wallCenter, out Vector3 inwardNormal))
        {
            Debug.LogWarning($"[CommandProcessor] near_wall: wall info missing for side={a.side}");
            return target.position;
        }

        float gap = (a.distance_m <= 0f) ? DEFAULT_GAP : a.distance_m;
        inwardNormal.Normalize();

        if (!TryGetWorldBounds(target, out Bounds tgtBounds))
        {
            float dist = roomController.GetWallThickness() * 0.5f + gap;
            Vector3 fallback = wallCenter + inwardNormal * dist;
            float floorY = roomController.GetFloorTopY();
            fallback.y = floorY;
            return fallback;
        }

        float tgtHalf = GetHalfSizeAlong(tgtBounds, -inwardNormal);
        float centerDistance = roomController.GetWallThickness() * 0.5f + gap + tgtHalf;

        Vector3 horizontal = wallCenter + inwardNormal * centerDistance;

        float floorTopY = roomController.GetFloorTopY();
        float objBottom = tgtBounds.min.y;
        float offsetY = floorTopY - objBottom;

        Vector3 finalPos = horizontal + new Vector3(0f, offsetY, 0f);
        return finalPos;
    }

    private Vector3 GetDirectionForRelation(string relation, string side, Transform reference)
    {

        if (!string.IsNullOrEmpty(side) && side != "none")
        {
    
            switch (side)
            {
                case "left":
                    return -FlattenXZ(reference.right);
                case "right":
                    return FlattenXZ(reference.right);
                case "front":
                case "in_front_of":
                    return FlattenXZ(reference.forward);
                case "back":
                case "behind":
                    return -FlattenXZ(reference.forward);
            }
        }

     
        switch (relation)
        {
            case "behind":
                return -FlattenXZ(reference.forward);

            case "in_front_of":
                return FlattenXZ(reference.forward);

            case "left_of":
                return -FlattenXZ(reference.right);

            case "right_of":
                return FlattenXZ(reference.right);

            case "on_top_of":
                return Vector3.up;

            case "under":
                return Vector3.down;

            case "extend_forward":
                return FlattenXZ(reference.forward);

            case "near":
            case "next_to":
            case "between":
               
                return FlattenXZ(reference.forward);

            case "inside":
                return Vector3.zero;

            default:
                return FlattenXZ(reference.forward);
        }
    }
    private Vector3 FlattenXZ(Vector3 v)
    {
        v.y = 0f;
        if (v.sqrMagnitude < 0.0001f)
            v = Vector3.forward;
        return v.normalized;
    }



    private bool TryGetWorldBounds(Transform root, out Bounds bounds)
    {
        var renderers = root.GetComponentsInChildren<Renderer>();
        if (renderers != null && renderers.Length > 0)
        {
            bounds = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++)
                bounds.Encapsulate(renderers[i].bounds);
            return true;
        }

        var colliders = root.GetComponentsInChildren<Collider>();
        if (colliders != null && colliders.Length > 0)
        {
            bounds = colliders[0].bounds;
            for (int i = 1; i < colliders.Length; i++)
                bounds.Encapsulate(colliders[i].bounds);
            return true;
        }

        bounds = new Bounds(root.position, Vector3.zero);
        return false;
    }

    private float GetHalfSizeAlong(Bounds b, Vector3 dir)
    {
        dir = dir.normalized;
        Vector3 ext = b.extents;
        Vector3 absDir = new Vector3(Mathf.Abs(dir.x), Mathf.Abs(dir.y), Mathf.Abs(dir.z));
        return Vector3.Dot(ext, absDir);
    }

    private void SnapToFloor(Transform target)
    {
        if (roomController == null)
            return;

        if (!TryGetWorldBounds(target, out Bounds objBounds))
            return;

        float floorTopY = roomController.GetFloorTopY();
        float bottomY = objBounds.min.y;
        float offsetY = floorTopY - bottomY;

        target.position += new Vector3(0f, offsetY, 0f);
    }
}
