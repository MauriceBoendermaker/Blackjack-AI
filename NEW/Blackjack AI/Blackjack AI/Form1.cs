using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json;

namespace BlackjackAI
{
    public partial class Form1 : Form
    {
        private HttpClient _client;
        private ComboBox monitorCombo;
        private Panel canvas;
        private Button pboxGenButton;
        private Button confirmButton;
        private Button startButton;
        private Button resetButton;
        private Label roundLabel;
        private Label dealerValueLabel;

        public Form1()
        {
            InitializeComponent();
            _client = new HttpClient();
            PopulateMonitors();
            this.Load += Form1_Load;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            DrawCanvas();
        }

        private async void ConfirmButton_Click(object sender, EventArgs e)
        {
            var response = await _client.PostAsync("http://localhost:5000/confirm_monitor", null);
            if (response.IsSuccessStatusCode)
            {
                MessageBox.Show("Monitor confirmed!");
                pboxGenButton.Enabled = true;
            }
        }

        private async void StartButton_Click(object sender, EventArgs e)
        {
            var response = await _client.PostAsync("http://localhost:5000/start", null);
            if (response.IsSuccessStatusCode)
            {
                MessageBox.Show("Started!");
            }
        }

        private async void ResetButton_Click(object sender, EventArgs e)
        {
            var response = await _client.PostAsync("http://localhost:5000/reset", null);
            if (response.IsSuccessStatusCode)
            {
                MessageBox.Show("Round reset!");
            }
        }

        private async void PopulateMonitors()
        {
            var response = await _client.GetAsync("http://localhost:5000/get_monitors");
            if (response.IsSuccessStatusCode)
            {
                var monitors = JsonConvert.DeserializeObject<List<string>>(await response.Content.ReadAsStringAsync());
                monitorCombo.Items.AddRange(monitors.ToArray());
            }
        }

        private void DrawCanvas()
        {
            var g = canvas.CreateGraphics();
            g.Clear(System.Drawing.Color.White);
            g.FillRectangle(System.Drawing.Brushes.Black, 50, 50, 150, 50);
            g.DrawString("Dealer", new System.Drawing.Font("Arial", 10), System.Drawing.Brushes.White, new System.Drawing.PointF(100, 75));
            for (int i = 0; i < 7; i++)
            {
                int x1 = 50 + i * (60 + 10);
                int y1 = 150;
                g.DrawRectangle(System.Drawing.Pens.Black, x1, y1, 60, 90);
                g.DrawString($"Player {i + 1}", new System.Drawing.Font("Arial", 10), System.Drawing.Brushes.Black, new System.Drawing.PointF(x1 + 20, y1 + 100));
            }
        }
    }

    public class BackgroundProcessor
    {
        // Implement your background processing logic here
    }
}
