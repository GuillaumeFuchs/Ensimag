namespace WindowsFormsApplication1
{
    partial class Form1
    {
        /// <summary>
        /// Variable nécessaire au concepteur.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Nettoyage des ressources utilisées.
        /// </summary>
        /// <param name="disposing">true si les ressources managées doivent être supprimées ; sinon, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Code généré par le Concepteur Windows Form

        /// <summary>
        /// Méthode requise pour la prise en charge du concepteur - ne modifiez pas
        /// le contenu de cette méthode avec l'éditeur de code.
        /// </summary>
        private void InitializeComponent()
        {
            this.N_Tb = new System.Windows.Forms.TextBox();
            this.T_Tb = new System.Windows.Forms.TextBox();
            this.sigma_Tb = new System.Windows.Forms.TextBox();
            this.r_Tb = new System.Windows.Forms.TextBox();
            this.S0_Tb = new System.Windows.Forms.TextBox();
            this.MC_Lab = new System.Windows.Forms.Label();
            this.go_Btn = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // N_Tb
            // 
            this.N_Tb.Location = new System.Drawing.Point(13, 13);
            this.N_Tb.Name = "N_Tb";
            this.N_Tb.Size = new System.Drawing.Size(100, 20);
            this.N_Tb.TabIndex = 0;
            // 
            // T_Tb
            // 
            this.T_Tb.Location = new System.Drawing.Point(13, 53);
            this.T_Tb.Name = "T_Tb";
            this.T_Tb.Size = new System.Drawing.Size(100, 20);
            this.T_Tb.TabIndex = 1;
            // 
            // sigma_Tb
            // 
            this.sigma_Tb.Location = new System.Drawing.Point(13, 95);
            this.sigma_Tb.Name = "sigma_Tb";
            this.sigma_Tb.Size = new System.Drawing.Size(100, 20);
            this.sigma_Tb.TabIndex = 2;
            // 
            // r_Tb
            // 
            this.r_Tb.Location = new System.Drawing.Point(13, 135);
            this.r_Tb.Name = "r_Tb";
            this.r_Tb.Size = new System.Drawing.Size(100, 20);
            this.r_Tb.TabIndex = 3;
            // 
            // S0_Tb
            // 
            this.S0_Tb.Location = new System.Drawing.Point(13, 182);
            this.S0_Tb.Name = "S0_Tb";
            this.S0_Tb.Size = new System.Drawing.Size(100, 20);
            this.S0_Tb.TabIndex = 4;
            // 
            // MC_Lab
            // 
            this.MC_Lab.AutoSize = true;
            this.MC_Lab.Location = new System.Drawing.Point(162, 59);
            this.MC_Lab.Name = "MC_Lab";
            this.MC_Lab.Size = new System.Drawing.Size(23, 13);
            this.MC_Lab.TabIndex = 5;
            this.MC_Lab.Text = "MC";
            // 
            // go_Btn
            // 
            this.go_Btn.Location = new System.Drawing.Point(165, 135);
            this.go_Btn.Name = "go_Btn";
            this.go_Btn.Size = new System.Drawing.Size(75, 23);
            this.go_Btn.TabIndex = 6;
            this.go_Btn.Text = "Go";
            this.go_Btn.UseVisualStyleBackColor = true;
            this.go_Btn.Click += new System.EventHandler(this.go_Btn_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 262);
            this.Controls.Add(this.go_Btn);
            this.Controls.Add(this.MC_Lab);
            this.Controls.Add(this.S0_Tb);
            this.Controls.Add(this.r_Tb);
            this.Controls.Add(this.sigma_Tb);
            this.Controls.Add(this.T_Tb);
            this.Controls.Add(this.N_Tb);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox N_Tb;
        private System.Windows.Forms.TextBox T_Tb;
        private System.Windows.Forms.TextBox sigma_Tb;
        private System.Windows.Forms.TextBox r_Tb;
        private System.Windows.Forms.TextBox S0_Tb;
        private System.Windows.Forms.Label MC_Lab;
        private System.Windows.Forms.Button go_Btn;
    }
}

