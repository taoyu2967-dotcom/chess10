using Microsoft.UI.Xaml;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;

namespace Chess10d
{
    public sealed partial class MainWindow : Window
    {
        private const int Port = 8787;
        private Process? _server;

        public MainWindow()
        {
            InitializeComponent();
            Title = "chess10d — 10×10 国际象棋变体";
            StartServer();
            WaitForPort(Port, 20000);
            Web.Source = new System.Uri($"http://127.0.0.1:{Port}/chess10.html");
            Closed += (_, _) =>
            {
                try
                {
                    if (_server is { HasExited: false })
                        _server.Kill(entireProcessTree: true);
                }
                catch
                {
                    // 服务进程已退出则忽略
                }
            };
        }

        /// <summary>从 exe 位置向上找仓库根（以 server/server.js 为锚点），找到就拉起 node 服务端。</summary>
        private void StartServer()
        {
            var root = FindRepoRoot(AppContext.BaseDirectory);
            var serverJs = root is null ? null : Path.Combine(root, "server", "server.js");
            if (serverJs is null || !File.Exists(serverJs))
                return; // 纯壳模式：连接外部已启动的服务

            _server = Process.Start(new ProcessStartInfo("node", $"\"{serverJs}\"")
            {
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
            });
        }

        private static string? FindRepoRoot(string start)
        {
            var dir = start;
            for (var i = 0; i < 8; i++)
            {
                var cand = Path.GetFullPath(Path.Combine(dir, ".."));
                if (File.Exists(Path.Combine(cand, "server", "server.js")))
                    return cand;
                dir = cand;
            }
            return null;
        }

        /// <summary>轮询服务端口就绪；超时则照常导航（WebView2 会显示连接失败页）。</summary>
        private static void WaitForPort(int port, int timeoutMs)
        {
            var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
            while (DateTime.UtcNow < deadline)
            {
                try
                {
                    using var c = new TcpClient();
                    if (c.ConnectAsync("127.0.0.1", port).Wait(300) && c.Connected)
                        return;
                }
                catch
                {
                    // 端口未就绪，继续等
                }
            }
        }
    }
}
