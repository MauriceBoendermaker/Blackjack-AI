namespace BlackjackAI
{
    partial class Form1
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.Text = "BlackJack AI";
            this.Size = new System.Drawing.Size(1000, 700);
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;

            // Monitor Selection
            var monitorLabel = new System.Windows.Forms.Label() { Text = "Select Monitor:", Left = 30, Top = 20, AutoSize = true };
            this.monitorCombo = new System.Windows.Forms.ComboBox() { Left = 150, Top = 20, Width = 200 };
            this.confirmButton = new System.Windows.Forms.Button() { Text = "Confirm", Left = 360, Top = 20, Width = 100 };
            this.confirmButton.Click += ConfirmButton_Click;

            // Player Box Generation
            this.pboxGenButton = new System.Windows.Forms.Button() { Text = "Generate Player Boxes", Left = 30, Top = 60, Width = 200, Enabled = false };
            this.startButton = new System.Windows.Forms.Button() { Text = "Start", Left = 30, Top = 100, Width = 200 };
            this.startButton.Click += StartButton_Click;

            // Reset Round
            this.resetButton = new System.Windows.Forms.Button() { Text = "Reset Round", Left = 30, Top = 140, Width = 200 };
            this.resetButton.Click += ResetButton_Click;

            // Labels
            this.roundLabel = new System.Windows.Forms.Label() { Text = "Round: 000", Left = 10, Top = 5, AutoSize = true, Font = new System.Drawing.Font("Arial", 14, System.Drawing.FontStyle.Bold) };
            this.dealerValueLabel = new System.Windows.Forms.Label() { Text = "Dealer has: ", Left = 800, Top = 5, AutoSize = true, Font = new System.Drawing.Font("Arial", 14, System.Drawing.FontStyle.Bold) };

            // Canvas
            this.canvas = new System.Windows.Forms.Panel() { Left = 30, Top = 200, Width = 920, Height = 400, BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle, BackColor = System.Drawing.Color.White };

            // Adding Controls
            this.Controls.Add(monitorLabel);
            this.Controls.Add(this.monitorCombo);
            this.Controls.Add(this.confirmButton);
            this.Controls.Add(this.pboxGenButton);
            this.Controls.Add(this.startButton);
            this.Controls.Add(this.resetButton);
            this.Controls.Add(this.roundLabel);
            this.Controls.Add(this.dealerValueLabel);
            this.Controls.Add(this.canvas);
        }
    }
}
