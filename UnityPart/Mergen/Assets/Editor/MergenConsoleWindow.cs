#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using System;
using System.Diagnostics;
using System.IO;
using System.Collections.Concurrent;
using System.Text;


public class MergenConsoleWindow : EditorWindow
{
    private Texture2D _logo;
    private Process _proc;
    private string _input = "create a chair";
    private Vector2 _scroll;
    private string _log = "";


    private string _aiEngineDir;      // .../Mergen/ai_engine
    private string _pythonExe;        // .../ai_engine/.venv/Scripts/python.exe
    private string _scriptPath;       // .../ai_engine/run_command_loop.py
    private string _jsonOutPath;      // .../Assets/StreamingAssets/next_command.json

    private readonly ConcurrentQueue<string> _pending = new();
    private double _nextFlushTime = 0;
    private bool _engineReady = false;

    [MenuItem("MERGEN/AI Console")]
    public static void ShowWindow() => GetWindow<MergenConsoleWindow>("MERGEN AI Console");

    private void OnEnable()
    {
        _logo = Resources.Load<Texture2D>("mergen_logo");
        var projectRoot = Directory.GetParent(Application.dataPath)?.FullName ?? "";
        _aiEngineDir = Path.Combine(projectRoot, "ai_engine");
        _pythonExe = Path.Combine(_aiEngineDir, ".venv", "Scripts", "python.exe");
        _scriptPath = Path.Combine(_aiEngineDir, "run_command_loop.py");


        var streaming = Path.Combine(Application.dataPath, "StreamingAssets");
        if (!Directory.Exists(streaming)) Directory.CreateDirectory(streaming);
        _jsonOutPath = Path.Combine(streaming, "next_command.json");
        EditorApplication.update += EditorTick;
        EditorApplication.quitting += StopEngine;
    }

    private void OnDisable()
    {
        EditorApplication.update -= EditorTick;
        EditorApplication.quitting -= StopEngine;
    }

    private void OnGUI()
    {
        DrawHeader();
        EditorGUILayout.LabelField("Paths", EditorStyles.boldLabel);
        _aiEngineDir = EditorGUILayout.TextField("ai_engine dir", _aiEngineDir);
        _pythonExe = EditorGUILayout.TextField("python.exe", _pythonExe);
        _scriptPath = EditorGUILayout.TextField("run_command_loop.py", _scriptPath);
        _jsonOutPath = EditorGUILayout.TextField("next_command.json", _jsonOutPath);

        EditorGUILayout.Space(8);

        using (new EditorGUILayout.HorizontalScope())
        {
            GUI.enabled = _proc == null;
            if (GUILayout.Button("Start AI Engine", GUILayout.Height(28)))
                StartEngine();

            GUI.enabled = _proc != null;
            if (GUILayout.Button("Stop", GUILayout.Height(28)))
                StopEngine();

            GUI.enabled = true;
        }

        EditorGUILayout.Space(8);

        EditorGUILayout.LabelField("Command", EditorStyles.boldLabel);
        using (new EditorGUILayout.HorizontalScope())
        {
            _input = EditorGUILayout.TextField(_input);
            GUI.enabled = (_proc != null) && !_proc.HasExited && _engineReady;
            if (GUILayout.Button("Send", GUILayout.Width(80)))
                SendCommand(_input);
            GUI.enabled = true;
        }

        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("Logs", EditorStyles.boldLabel);

        _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.Height(260));
        EditorGUILayout.TextArea(_log, GUILayout.ExpandHeight(true));
        EditorGUILayout.EndScrollView();

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Clear Logs")) _log = "";
            if (GUILayout.Button("Reveal JSON"))
            {
                if (File.Exists(_jsonOutPath))
                    EditorUtility.RevealInFinder(_jsonOutPath);
                else
                    AppendLog("[Unity] JSON file not found yet.");
            }
        }
    }
    private void DrawHeader()
    {
        if (_logo == null) return;

        GUILayout.Space(8);

       
        var rect = GUILayoutUtility.GetRect(
            0,
            120,
            GUILayout.ExpandWidth(true)
        );

        float aspect = (float)_logo.width / _logo.height;
        float height = rect.height;
        float width = height * aspect;

        var centered = new Rect(
            rect.x + (rect.width - width) / 2f,
            rect.y,
            width,
            height
        );

        GUI.DrawTexture(centered, _logo, ScaleMode.ScaleToFit, true);
    }

    private void EditorTick()
    {
        if (EditorApplication.timeSinceStartup < _nextFlushTime) return;
        _nextFlushTime = EditorApplication.timeSinceStartup + 0.1; 

        if (_pending.IsEmpty) return;

        var sb = new StringBuilder();
        int n = 0;
        while (n < 200 && _pending.TryDequeue(out var line)) 
        {
            sb.AppendLine(line);

          
            if (!_engineReady && line.Contains("[READY]"))
                _engineReady = true;

            n++;
        }

        _log += sb.ToString();

        const int maxChars = 20000;
        if (_log.Length > maxChars)
            _log = _log.Substring(_log.Length - maxChars);

        Repaint();
    }


    private void StartEngine()
    {
        if (_proc != null) return;
        _engineReady = false;


        if (!File.Exists(_pythonExe))
        {
            AppendLog("[Unity] python.exe not found. Did you create .venv in ai_engine?");
            return;
        }
        if (!File.Exists(_scriptPath))
        {
            AppendLog("[Unity] run_command_loop.py not found.");
            return;
        }

        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            WorkingDirectory = _aiEngineDir,

            // -u = unbuffered 
            Arguments = $"-u -X utf8 \"{_scriptPath}\" --json-out \"{_jsonOutPath}\"",
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true


        };
        psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
        psi.StandardOutputEncoding = System.Text.Encoding.UTF8;
        psi.StandardErrorEncoding = System.Text.Encoding.UTF8;

        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

        _proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) AppendLog(e.Data); };
        _proc.ErrorDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) AppendLog("[ERR] " + e.Data); };
        _proc.Exited += (_, __) =>
        {
            _pending.Enqueue($"[Unity] AI Engine exited. ExitCode={_proc.ExitCode}");
            _engineReady = false;
        };


        try
        {
            _proc.Start();
            _proc.BeginOutputReadLine();
            _proc.BeginErrorReadLine();
            AppendLog("[Unity] AI Engine started.");
        }
        catch (Exception ex)
        {
            AppendLog("[Unity] Failed to start: " + ex.Message);
            _proc = null;
        }
    }

    private void SendCommand(string cmd)
    {
        if (_proc == null || _proc.HasExited) { AppendLog("[Unity] Engine not running."); return; }
        if (string.IsNullOrWhiteSpace(cmd)) return;

        try
        {
            _proc.StandardInput.WriteLine(cmd);
            _proc.StandardInput.Flush();
        }
        catch (Exception ex)
        {
            AppendLog("[Unity] Send failed: " + ex.Message);
        }
    }

    private void StopEngine()
    {
        if (_proc == null) return;

        try
        {
            if (!_proc.HasExited)
            {
                _proc.StandardInput.WriteLine("q");
                _proc.StandardInput.Flush();
            }
        }
        catch { /* ignore */ }
        try
        {
            _proc.WaitForExit(800);
        }
        catch { /* ignore */ }
        try
        {
            if (!_proc.HasExited)
                _proc.Kill();
        }
        catch { /* ignore */ }

        try { _proc.Dispose(); } catch { /* ignore */ }
        _proc = null;

        AppendLog("[Unity] AI Engine stopped.");
    }

    private void AppendLog(string line)
    {
        _pending.Enqueue(line);
    }
}
#endif
