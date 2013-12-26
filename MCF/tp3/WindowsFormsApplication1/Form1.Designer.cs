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
            this.ICMC_Lab = new System.Windows.Forms.Label();
            this.PrixMCC_Lab = new System.Windows.Forms.Label();
            this.ICMCC_Lab = new System.Windows.Forms.Label();
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
            // ICMC_Lab
            // 
            this.ICMC_Lab.AutoSize = true;
            this.ICMC_Lab.Location = new System.Drawing.Point(12, 50);
            this.ICMC_Lab.Name = "ICMC_Lab";
            this.ICMC_Lab.Size = new System.Drawing.Size(38, 13);
            this.ICMC_Lab.TabIndex = 7;
            this.ICMC_Lab.Text = "Ic_MC";
            // 
            // PrixMCC_Lab
            // 
            this.PrixMCC_Lab.AutoSize = true;
            this.PrixMCC_Lab.Location = new System.Drawing.Point(139, 9);
            this.PrixMCC_Lab.Name = "PrixMCC_Lab";
            this.PrixMCC_Lab.Size = new System.Drawing.Size(53, 13);
            this.PrixMCC_Lab.TabIndex = 8;
            this.PrixMCC_Lab.Text = "Prix_MCC";
            // 
            // ICMCC_Lab
            // 
            this.ICMCC_Lab.AutoSize = true;
            this.ICMCC_Lab.Location = new System.Drawing.Point(142, 50);
            this.ICMCC_Lab.Name = "ICMCC_Lab";
            this.ICMCC_Lab.Size = new System.Drawing.Size(45, 13);
            this.ICMCC_Lab.TabIndex = 9;
            this.ICMCC_Lab.Text = "Ic_MCC";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 262);
            this.Controls.Add(this.ICMCC_Lab);
            this.Controls.Add(this.PrixMCC_Lab);
            this.Controls.Add(this.ICMC_Lab);
            this.Controls.Add(this.PrixMC_Lab);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label PrixMC_Lab;
        private System.Windows.Forms.Label ICMC_Lab;
        private System.Windows.Forms.Label PrixMCC_Lab;
        private System.Windows.Forms.Label ICMCC_Lab;
    }
}

