<%@ Page Title="Page d'accueil" Language="C#" MasterPageFile="~/Site.master" AutoEventWireup="true"
    CodeBehind="Default.aspx.cs" Inherits="Interface._Default" %>

<asp:Content ID="HeaderContent" runat="server" ContentPlaceHolderID="HeadContent">
</asp:Content>
<asp:Content ID="BodyContent" runat="server" ContentPlaceHolderID="MainContent">
    <h2>
        Bienvenue dans ASP.NET!
    </h2>
    <p>
        Pour en savoir plus sur ASP.NET, consultez <a href="http://www.asp.net" title="Site Web ASP.NET">www.asp.net</a>.
    </p>
    <p>
        Vous pouvez également trouver de la <a href="http://go.microsoft.com/fwlink/?LinkID=152368"
            title="Documentation ASP.NET sur MSDN">documentation sur ASP.NET sur MSDN</a>.
    </p>
    <p>
        <asp:Label id="strike_label" text="strike" runat="server"/>
        <asp:TextBox id="strike_tb" runat="server"/>
    </p>
    <p>
        <asp:Label id="sigma_label" text="sigma" runat="server"/>
        <asp:TextBox id="sigma_tb" runat="server"/>
    </p>
    <p>
        <asp:Label id="size_label" text="size" runat="server"/>
        <asp:TextBox ID="size_tb" runat="server"></asp:TextBox>
    </p>
    <p>
        <asp:Label id="r_label" text="r" runat="server"/>
        <asp:TextBox ID="r_tb" runat="server"></asp:TextBox>
   </p>
    <p>
        <asp:Label id="rho_label" text="rho" runat="server"/>
        <asp:TextBox ID="rho_tb" runat="server"></asp:TextBox>
    </p>
    <p>
        <asp:Label id="h_label" text="h" runat="server"/>
        <asp:TextBox ID="h_tb" runat="server"></asp:TextBox>
    </p>
    <p>
        <asp:Label id="samples_label" text="samples" runat="server"/>
        <asp:TextBox ID="samples_tb" runat="server"></asp:TextBox>
    </p>
    <p>
        <asp:Label id="prix_label" runat="server">Prix</asp:Label>
    </p>
    <p>
        <asp:Label id="ic_label" runat="server">IC</asp:Label>
    </p>
    <p>
    <asp:Button id="gebtn1" text="Start" OnClick='price' runat="server"/>
    </p>
</asp:Content>
