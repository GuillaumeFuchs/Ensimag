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
            this.PrixMC_Lab = new System.Windows.Forms.Label();
            this.go_Btn = new System.Windows.Forms.Button();
            this.ICMC_Lab = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // PrixMC_Lab
            // 
            this.PrixMC_Lab.AutoSize = true;
            this.PrixMC_Lab.Location = new System.Drawing.Point(12, 9);
            this.PrixMC_Lab.Name = "PrixMC_Lab";
            this.PrixMC_Lab.Size = new System.Drawing.Size(46, 13);
            this.PrixMC_Lab.TabIndex = 5;
            this.PrixMC_Lab.Text = "Prix_MC";
            // 
            // go_Btn
            // 
            this.go_Btn.Location = new System.Drawing.Point(96, 169);
            this.go_Btn.Name = "go_Btn";
            this.go_Btn.Size = new System.Drawing.Size(75, 23);
            this.go_Btn.TabIndex = 6;
            this.go_Btn.Text = "Go";
            this.go_Btn.UseVisualStyleBackColor = true;
            this.go_Btn.Click += new System.EventHandler(this.go_Btn_Click);
            // 
            // ICMC_Lab
            // 
            this.ICMC_Lab.AutoSize = true;
            this.ICMC_Lab.Location = new System.Drawing.Point(12, 50);
            this.ICMC_Lab.Name = "ICMC_Lab";
            this.ICMC_Lab.Size = new System.Drawing.Size(38, 13);
            this.ICMC_Lab.TabIndex = 7;
            this.ICMC_Lab.Text = "Ic_MC";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 262);
            this.Controls.Add(this.ICMC_Lab);
            this.Controls.Add(this.go_Btn);
            this.Controls.Add(this.PrixMC_Lab);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label PrixMC_Lab;
        private System.Windows.Forms.Button go_Btn;
        private System.Windows.Forms.Label ICMC_Lab;
    }
}

