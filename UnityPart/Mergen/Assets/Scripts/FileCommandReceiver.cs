using System;
using System.IO;
using UnityEngine;

public class FileCommandReceiver : MonoBehaviour
{
    [Tooltip("How often (in seconds) to poll the file?")]
    public float pollInterval = 0.2f;

    [Tooltip("Name of the JSON file inside StreamingAssets")]
    public string commandFileName = "next_command.json";


    public event Action<CommandMessage> OnCommandReceived;

    private float _timer;

    private void Update()
    {
        _timer += Time.deltaTime;
        if (_timer >= pollInterval)
        {
            _timer = 0f;
            TryReadCommand();
        }
    }

    private void TryReadCommand()
    {
        string path = Path.Combine(Application.streamingAssetsPath, commandFileName);

        if (!File.Exists(path))
            return;

        string json = File.ReadAllText(path);
        if (string.IsNullOrWhiteSpace(json))
            return;

        try
        {
            CommandMessage cmd = JsonUtility.FromJson<CommandMessage>(json);
            if (cmd != null)
            {
                Debug.Log($"[CommandReceiver] New command: {cmd.intent}");
                OnCommandReceived?.Invoke(cmd);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[CommandReceiver] JSON parse error: {e.Message}");
        }


        File.Delete(path);
    }
}