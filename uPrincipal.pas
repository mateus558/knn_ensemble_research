unit uPrincipal;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, Buttons, Grids, Math, ComCtrls, ExtCtrls, TeEngine, Series,
  TeeProcs, Chart, ComObj, Excel2000;//, scExcelExport;

type
  Vetor8 = array [0..7] of Integer;

type
  TfrmPrincipal = class(TForm)
    Label7: TLabel;
    Timer1: TTimer;
    lbData: TLabel;
    PageControl1: TPageControl;
    Unidimensionais: TTabSheet;
    Label2: TLabel;
    Label5: TLabel;
    Label15: TLabel;
    Label16: TLabel;
    Label17: TLabel;
    Label18: TLabel;
    Label19: TLabel;
    Label20: TLabel;
    Label21: TLabel;
    Label22: TLabel;
    Label23: TLabel;
    Label24: TLabel;
    Label25: TLabel;
    lbSolAlePen: TLabel;
    lbMelhorSol: TLabel;
    Label1: TLabel;
    Label3: TLabel;
    Label6: TLabel;
    Label11: TLabel;
    Label26: TLabel;
    Label27: TLabel;
    lbTime: TLabel;
    Label28: TLabel;
    Label4: TLabel;
    Label29: TLabel;
    Label30: TLabel;
    Label31: TLabel;
    Label32: TLabel;
    strgProxEst: TStringGrid;
    edtTempInicial: TEdit;
    GeraSA: TBitBtn;
    edtAlfa: TEdit;
    strgACUni: TStringGrid;
    BitBtn1: TBitBtn;
    strgRegra1: TStringGrid;
    strgRegra2: TStringGrid;
    strgRegra3: TStringGrid;
    strgRegra4: TStringGrid;
    strgRegra5: TStringGrid;
    strgRegra6: TStringGrid;
    strgRegra7: TStringGrid;
    strgRegra8: TStringGrid;
    nentropia: TEdit;
    strgMelhor: TStringGrid;
    strgEntropia: TStringGrid;
    chrtGraficoEntropia: TChart;
    Series1: TLineSeries;
    chrtGraficoEntropiaMedia: TChart;
    AreaSeries1: TLineSeries;
    edtTempAtual: TEdit;
    edtMediaEnt: TEdit;
    edtMediaEnt100Ant: TEdit;
    edtEntropia: TEdit;
    edtDifMedias: TEdit;
    strgMelhoresRegras: TStringGrid;
    edtIteracao: TEdit;
    edtMediaTempAnterior: TEdit;
    strgRegrasFixas: TStringGrid;
    rgRegras: TRadioGroup;
    edtMinIterT: TEdit;
    edtEpslon: TEdit;
    edtTempMinima: TEdit;
    strgACRegraUni: TStringGrid;
    Sobre: TTabSheet;
    Label10: TLabel;
    Label9: TLabel;
    Label8: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    btnExportar: TBitBtn;
    strgACInicial: TStringGrid;
    lbACInicial: TLabel;
    btnExpAle: TBitBtn;
    btnExpMelhor: TBitBtn;
    procedure GeraSAClick(Sender: TObject);
    procedure strgrdNovoAC1DrawCell(Sender: TObject; ACol, ARow: Integer;
      Rect: TRect; State: TGridDrawState);
    procedure BitBtn1Click(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure rgRegrasClick(Sender: TObject);
    procedure btnExportarClick(Sender: TObject);
    procedure btnExpAleClick(Sender: TObject);
    procedure btnExpMelhorClick(Sender: TObject);
  private
    { Private declarations }
    function IntToBin8(valor: Integer): vetor8;
    function BinToInt8(valor: vetor8): Integer;
    function Vizinhanca8(Col, Row: Integer): vetor8;
    function Entropia: Real;
    procedure SA(Decrescimo: Real; TInicial: Real; TempMinima: Real; MinIterT: Integer; Epslon: Real);
    procedure CalculaAutomato(regraAleatoria: Integer;RegraBinAle: vetor8);
  public
    { Public declarations }
  end;

var
  frmPrincipal: TfrmPrincipal;
  teste: String;
  Ordem, Global, Global8: Integer;
  vetEntropia: array[0..60000] of Real;
  gEnt : Integer;
  melhor : Integer;
  RegraUniInt: Vetor8;
  ValoresAceitos: array of Double;
  vetMedias: array of Double;
  iAtual: Integer;
  SomaAtual: Real;
  MediaAtual: Real;
  Media100Ant: Real;
  MediaTempAnterior: Real;
  iUltima: Integer;
  vetEntropiaTempAnterior: array [0..7] of Real;
  EntropiaTempAnterior: Real;
  TempoInicio: TTime;
  RegraUniBin1, RegraUniBin2, RegraUniBin3, RegraUniBin4: Vetor8;
  RegraUniBin5, RegraUniBin6, RegraUniBin7, RegraUniBin8: Vetor8;
  IteracoesTotal: Integer;
  MenorMedia, MaiorMedia: Real;
  Maiores78: Integer;


implementation

{$R *.dfm}

function tFrmPrincipal.Entropia: Real;
var
  Vetor: array [0..255] of Integer;
  Col, Row, i, qq: Integer;
  p: Real;
  linha: Vetor8;
begin
  for i := 0 to 255 do
     vetor[i] := 0;

  for Row := 1 to (strgACUni.RowCount - 1) do
  begin
    for Col := 0 to (strgACUni.ColCount - 1)do
    begin
      linha[Col] := StrToInt(strgACUni.Cells[Col, Row]);
    end;
    qq := BinToInt8(linha);
    vetor[qq] := vetor[qq] + 1;
  end;

  result := 0;
  for i := 0 to 255 do
  begin
    p := vetor[i]/(StrToInt(nEntropia.text));
    if not (p = 0) then
      p := p * log2(p);
    result := result - p;
  end;
end;

procedure tFrmPrincipal.SA(Decrescimo: Real; TInicial: Real; TempMinima: Real; MinIterT: Integer; Epslon: Real);
var
  T, probAleatorio, delta, boltzmann: Real;
  regraAleatoria: Integer;
  RegraBinAle: Vetor8;
  i,j: integer;
  IterT: Integer;
  DifMedias: Real;
begin
  T := TInicial; {Temperatura corrente}
  Randomize;
  strgMelhor.RowCount := strgACUni.RowCount;
  for i:= 0 to (strgACUni.RowCount - 1)do
    for j:= 0 to (strgACUni.ColCount - 1)do
      strgMelhor.Cells[j,i] := strgACUni.cells[j,i];
  iUltima := 0;
  MediaTempAnterior := 0;
  for i:=0 to 7 do
    vetEntropiaTempAnterior[i] := 0;
  EntropiaTempAnterior := 0;
  IteracoesTotal := 0; {Número de iterações total}
  Maiores78 := 0;
  while T > TempMinima do {"correto" seria while MediaAtual > MediaTempAnterior do}
  begin
    edtTempAtual.Text := FloatToStr(T);
    IterT := 0; {Número de iterações na temperatura atual}
    DifMedias := 1;
    iAtual := 1;
    SomaAtual := ValoresAceitos[Length(ValoresAceitos) - 1];
    while DifMedias > Epslon do
    begin
      MaiorMedia := 0;
      MenorMedia := 8;
      inc(IterT);
      inc(IteracoesTotal);
      edtIteracao.Text := IntToStr(IterT);
      regraAleatoria := random(8);
      RegraBinAle := IntToBin8(Random(256));
      CalculaAutomato(regraAleatoria, RegraBinAle);

      delta := vetEntropia[gEnt-1] - vetEntropia[melhor];

      strgEntropia.Cells[gEnt-1,2] := IntToStr(IteracoesTotal);

      Media100Ant := 0;
      if (delta >= 0) then
      begin
        for i:=0 to 7 do
          for j:=0 to strgACUni.RowCount-1 do
            strgMelhor.Cells[i, j] := strgACUni.Cells[i, j];
        strgACRegraUni.Cells[regraAleatoria,0] := IntToStr(BinToInt8(RegraBinAle));
        if delta = 0 then
          strgEntropia.Cells[gEnt-1,1] := '3'
        else
          strgEntropia.Cells[gEnt-1,1] := '1';

        if vetEntropia[gEnt-1] > 7.8 then
        begin
          strgMelhoresRegras.RowCount := Maiores78 + 1;
          for i := 0 to 7 do
            strgMelhoresRegras.Cells[i, Maiores78] := strgACRegraUni.Cells[i,0];
          strgMelhoresRegras.Cells[8, Maiores78] := FloatToStr(vetEntropia[gEnt-1]);
          inc(Maiores78);
        end;

        for i := 0 to 7 do
        begin
          vetEntropiaTempAnterior[i] := StrToFloat(strgACRegraUni.Cells[i,0]);
        end;
        melhor := gEnt-1;

        EntropiaTempAnterior := vetEntropia[melhor];
        SetLength(ValoresAceitos, Length(ValoresAceitos) + 1);
        ValoresAceitos[Length(ValoresAceitos) - 1] := StrToFloat(strgEntropia.Cells[melhor, 0]);
        chrtGraficoEntropia.Series[0].Add(ValoresAceitos[Length(ValoresAceitos) - 1]);
        edtEntropia.Text := FloatToStr(ValoresAceitos[Length(ValoresAceitos) - 1]);

        inc(iAtual);// := Length(ValoresAceitos);
//        edtiAtual.Text := IntToStr(iAtual);

        SomaAtual := ValoresAceitos[Length(ValoresAceitos) - 1] + SomaAtual;
        MediaAtual := SomaAtual / iAtual;
        edtMediaEnt.text := FloatToStr(MediaAtual);
        SetLength(vetMedias, Length(vetMedias) + 1);
        vetMedias[Length(ValoresAceitos) - 1] := MediaAtual;

        chrtGraficoEntropiaMedia.Series[0].Add(MediaAtual);
        Application.ProcessMessages;

        if (IterT > MinIterT) then
        begin
          if (MediaTempAnterior - MediaAtual) > 0.5 then
            edtEntropia.Text := FloatToStr(EntropiaTempAnterior);

          MediaTempAnterior := MediaAtual;
          edtMediaTempAnterior.Text := FloatToStr(MediaTempAnterior);
          Media100Ant := vetMedias[IteracoesTotal - MinIterT];
          i := IteracoesTotal;
          while i <> (IteracoesTotal - MinIterT) do     //Diferença entre a média máxima e a mínima dentre as mínimas iterações por temperatura
          begin
            if vetMedias[i] > MaiorMedia then
              MaiorMedia := vetMedias[i];
            if vetMedias[i] < MenorMedia then
              MenorMedia := vetMedias[i];
            dec(i);
          end;
          DifMedias := (MaiorMedia - MenorMedia);
        end;
        edtMediaEnt100Ant.Text := FloatToStr(Media100Ant);
        edtDifMedias.Text := FloatToStr(DifMedias);
      end
      else //delta < 0
      begin
        strgEntropia.Cells[gEnt-1,1] := '0';
        probAleatorio := random;
        boltzmann := exp(delta/T); {Distribuição de Boltzmann}
        if (probAleatorio < boltzmann) then
        begin
          for i:=0 to 7 do
            for j:=0 to strgACUni.RowCount-1 do
              strgMelhor.Cells[i, j] := strgACUni.Cells[i, j];
          strgACRegraUni.Cells[regraAleatoria,0] := IntToStr(BinToInt8(RegraBinAle));
          strgEntropia.Cells[gEnt-1,1] := '2';

          if vetEntropia[gEnt-1] >= 7.8 then
          begin
            strgMelhoresRegras.RowCount := Maiores78 + 1;
            for i := 0 to 7 do
              strgMelhoresRegras.Cells[i, Maiores78] := strgACRegraUni.Cells[i,0];
            strgMelhoresRegras.Cells[8, Maiores78] := FloatToStr(vetEntropia[gEnt-1]);
            inc(Maiores78);
          end;

          for i := 0 to 7 do
            vetEntropiaTempAnterior[i] := StrToFloat(strgACRegraUni.Cells[i,0]);

          melhor := gEnt-1;
          EntropiaTempAnterior := vetEntropia[melhor];
          SetLength(ValoresAceitos, Length(ValoresAceitos) + 1);
          ValoresAceitos[Length(ValoresAceitos) - 1] := StrToFloat(strgEntropia.Cells[melhor, 0]);
          chrtGraficoEntropia.Series[0].Add(ValoresAceitos[Length(ValoresAceitos) - 1]);
          edtEntropia.Text := FloatToStr(ValoresAceitos[Length(ValoresAceitos) - 1]);

          inc(iAtual);// := Length(ValoresAceitos);
//          edtiAtual.Text := IntToStr(iAtual);

          SomaAtual := ValoresAceitos[Length(ValoresAceitos) - 1] + SomaAtual;
          MediaAtual := SomaAtual / iAtual;
          edtMediaEnt.text := FloatToStr(MediaAtual);
          SetLength(vetMedias, Length(vetMedias) + 1);
          vetMedias[Length(ValoresAceitos) - 1] := MediaAtual;

          chrtGraficoEntropiaMedia.Series[0].Add(MediaAtual);

          if (IterT > MinIterT) then
          begin
            if (MediaTempAnterior - MediaAtual) > 0.5 then
            begin
              edtEntropia.Text := FloatToStr(EntropiaTempAnterior);

            end;
            MediaTempAnterior := MediaAtual;
            edtMediaTempAnterior.Text := FloatToStr(MediaTempAnterior);
            Media100Ant := vetMedias[IteracoesTotal - MinIterT];
            i := IteracoesTotal;
            while i <> (IteracoesTotal - MinIterT) do     //Diferença entre a média máxima e a mínima dentre as mínimas iterações por temperatura
            begin
              if vetMedias[i] > MaiorMedia then
                MaiorMedia := vetMedias[i];
              if vetMedias[i] < MenorMedia then
                MenorMedia := vetMedias[i];
              dec(i);
            end;
            DifMedias := (MaiorMedia - MenorMedia);
          end;
          edtMediaEnt100Ant.Text := FloatToStr(Media100Ant);
          edtDifMedias.Text := FloatToStr(DifMedias);
          Application.ProcessMessages;
        end // probAleatorio < boltzmann
        else // probAleatorio >= boltzmann
        begin
          for i := 0 to 7 do
            RegraUniInt[I] := StrToInt(strgACRegraUni.Cells[I,0]);

          RegraUniBin1 := IntToBin8(RegraUniInt[0]);
          RegraUniBin2 := IntToBin8(RegraUniInt[1]);
          RegraUniBin3 := IntToBin8(RegraUniInt[2]);
          RegraUniBin4 := IntToBin8(RegraUniInt[3]);
          RegraUniBin5 := IntToBin8(RegraUniInt[4]);
          RegraUniBin6 := IntToBin8(RegraUniInt[5]);
          RegraUniBin7 := IntToBin8(RegraUniInt[6]);
          RegraUniBin8 := IntToBin8(RegraUniInt[7]);

          for I := 0 to 7 do
          begin
            strgRegra1.Cells[0,i] := IntToStr(i);
            strgRegra1.Cells[1,i] := IntToStr(RegraUniBin1[i]);
            strgRegra2.Cells[0,i] := IntToStr(i);
            strgRegra2.Cells[1,i] := IntToStr(RegraUniBin2[i]);
            strgRegra3.Cells[0,i] := IntToStr(i);
            strgRegra3.Cells[1,i] := IntToStr(RegraUniBin3[i]);
            strgRegra4.Cells[0,i] := IntToStr(i);
            strgRegra4.Cells[1,i] := IntToStr(RegraUniBin4[i]);
            strgRegra5.Cells[0,i] := IntToStr(i);
            strgRegra5.Cells[1,i] := IntToStr(RegraUniBin5[i]);
            strgRegra6.Cells[0,i] := IntToStr(i);
            strgRegra6.Cells[1,i] := IntToStr(RegraUniBin6[i]);
            strgRegra7.Cells[0,i] := IntToStr(i);
            strgRegra7.Cells[1,i] := IntToStr(RegraUniBin7[i]);
            strgRegra8.Cells[0,i] := IntToStr(i);
            strgRegra8.Cells[1,i] := IntToStr(RegraUniBin8[i]);
          end;

          SetLength(ValoresAceitos, Length(ValoresAceitos) + 1);
          ValoresAceitos[Length(ValoresAceitos) - 1] := StrToFloat(strgEntropia.Cells[melhor, 0]);
//          chrtGraficoEntropia.Series[0].Add(ValoresAceitos[Length(ValoresAceitos) - 1]);
          edtEntropia.Text := FloatToStr(ValoresAceitos[Length(ValoresAceitos) - 1]);

          inc(iAtual);// := Length(ValoresAceitos);
//          edtiAtual.Text := IntToStr(iAtual);

          SomaAtual := ValoresAceitos[Length(ValoresAceitos) - 1] + SomaAtual;
          MediaAtual := SomaAtual / iAtual;
          edtMediaEnt.text := FloatToStr(MediaAtual);
          SetLength(vetMedias, Length(vetMedias) + 1);
          vetMedias[Length(ValoresAceitos) - 1] := MediaAtual;

//          chrtGraficoEntropiaMedia.Series[0].Add(MediaAtual);
          Application.ProcessMessages;

          if (IterT > MinIterT) then
          begin
            if (MediaTempAnterior - MediaAtual) > 0.5 then
              edtEntropia.Text := FloatToStr(EntropiaTempAnterior);

            MediaTempAnterior := MediaAtual;
            edtMediaTempAnterior.Text := FloatToStr(MediaTempAnterior);
            Media100Ant := vetMedias[IteracoesTotal - MinIterT];
            i := IteracoesTotal;
            while i <> (IteracoesTotal - MinIterT) do     //Diferença entre a média máxima e a mínima dentre as mínimas iterações por temperatura
            begin
              if vetMedias[i] > MaiorMedia then
                MaiorMedia := vetMedias[i];
              if vetMedias[i] < MenorMedia then
                MenorMedia := vetMedias[i];
              dec(i);
            end;
            DifMedias := (MaiorMedia - MenorMedia);
          end;
          edtMediaEnt100Ant.Text := FloatToStr(Media100Ant);
          edtDifMedias.Text := FloatToStr(DifMedias);

        end; // probAleatorio >= boltzmann
      end;
    end;
    T := Decrescimo * T;
//    IterT := 0;
  end;
end;


procedure TfrmPrincipal.GeraSAClick(Sender: TObject);
var
  Decrescimo, T0, TempMinima, Epslon: Real;
  MinIterT: Integer;
begin

  btnExpMelhor.Visible := true;

  TempoInicio := Time;
  Timer1.Enabled := true;

  Decrescimo := StrToFloat(edtAlfa.Text);
  T0 := StrToFloat(edtTempInicial.Text);
  TempMinima := StrToFloat(edtTempMinima.Text);
  MinIterT := StrToInt(edtMinIterT.Text);
  Epslon := StrToFloat(edtEpslon.Text);

  SA(Decrescimo, T0, TempMinima, MinIterT, Epslon);

  Timer1.Enabled := false;
end;

function TfrmPrincipal.IntToBin8(valor: Integer): vetor8;
var
  i,j: Integer;
  Parar: Boolean;
  temp: vetor8;
begin
  i:= 0;
  parar:= false;
  while (parar = false) do
  begin
    if ((valor mod 2) = 1) then
    begin
      temp[i]:= 1;
      i:= i + 1;
    end
    else
    begin
      temp[i]:= 0;
      i:= i + 1;
    end;
    valor:= valor div 2;
    if (valor <1) then
      parar:= true;
  end;
  for j:=i to 7 do
    temp[j]:= 0;
  result:= temp;
end;

function TfrmPrincipal.BinToInt8(valor: vetor8): Integer;
var
  i, j: Integer;
begin
  Result := 0;
  j := 1;
  i:= 7;
  while (i > -1) do
  begin
    Result := Result + (valor[i])*j;
    j := j*2;
    i:= i - 1;
  end;
end;

function TfrmPrincipal.Vizinhanca8(Col, Row: integer): vetor8;
var
  k: integer;
begin
  //Oeste
  if (Col = 0) then
    Vizinhanca8[5]:= StrToInt(strgACUni.Cells[7,Row])
  else
    Vizinhanca8[5]:= StrToInt(strgACUni.Cells[Col - 1,Row]);
  //Centro
  Vizinhanca8[6]:= StrToInt(strgACUni.Cells[Col,Row]);
  //Leste
  if (Col = 7) then
    Vizinhanca8[7]:= StrToInt(strgACUni.Cells[0,Row])
  else
    Vizinhanca8[7]:= StrToInt(strgACUni.Cells[Col + 1,Row]);
  //Preencher o resto com 0 (zero)
  for k:= 4 downto 0 do
    Vizinhanca8[k]:= 0;

  strgProxEst.cells[row,global8]:= IntToStr(BinToInt8(result));
  global8:= global8 + 1;
end;

procedure TfrmPrincipal.strgrdNovoAC1DrawCell(Sender: TObject; ACol,
  ARow: Integer; Rect: TRect; State: TGridDrawState);
begin
  with TStringGrid(Sender) do
  begin
    if Cells[ACol, ARow] = '0' then
    begin
      Canvas.Brush.Color:=clWhite;
      Canvas.FillRect(Rect);
      Canvas.Font.Color:=clWhite;
      Canvas.TextOut(Rect.Left, Rect.Top, Cells[ACol, ARow]);
    end
    else if Cells[ACol, ARow] = '1' then
    begin
      Canvas.Brush.Color:=clBlack;
      Canvas.FillRect(Rect);
      Canvas.Font.Color:=clBlack;
      Canvas.TextOut(Rect.Left, Rect.Top, Cells[ACol, ARow]);
    end;
  end;

end;

procedure TfrmPrincipal.BitBtn1Click(Sender: TObject);
var
  Primeiro: vetor8;
begin
  melhor := 0;
  if rgRegras.ItemIndex = 0 then
    CalculaAutomato(8,Primeiro) // Solução inicial
  else if rgRegras.ItemIndex = 1 then
    CalculaAutomato(9,Primeiro); // Regras fixas

 { //#13 = enter e #8 = BackSpace
  if not (key in ['1','0',#13,#8]) then begin
   showmessage('Digite 0 ou 1');
 end;}

end;

procedure TfrmPrincipal.CalculaAutomato(regraAleatoria: Integer;RegraBinAle: vetor8);
var
  I, Row, Col, n: Integer;
begin
  Randomize;
  if (regraAleatoria = 8) then // Solução inicial
  begin
    gEnt:= 0;
    Global8 := 0;
    strgProxEst.ColCount := StrToInt(nentropia.Text);
    strgACUni.RowCount := StrToInt(nentropia.Text) + 1;
    for I := 0 to 7 do
    begin
      strgACUni.Cells[I,0] := IntToStr(Random(2));
      strgACRegraUni.Cells[I,0] := IntToStr(Random(256));
      RegraUniInt[I] := StrToInt(strgACRegraUni.Cells[I,0]);
    end;
  end
  else if (regraAleatoria = 9) then // Regras fixas
  begin
    gEnt:= 0;
    Global8 := 0;
    strgProxEst.ColCount := StrToInt(nentropia.Text);
    strgACUni.RowCount := StrToInt(nentropia.Text) + 1;
    for I := 0 to 7 do
    begin
      if (strgACInicial.Cells[I,0] <> '0') and (strgACInicial.Cells[I,0] <> '1') then // Autômato inicial definido
        strgACUni.Cells[I,0] := IntToStr(Random(2))  // Autômato inicial aleatório
      else
        strgACUni.Cells[I,0] := strgACInicial.Cells[I,0];
      RegraUniInt[I] := StrToInt(strgRegrasFixas.Cells[I,0]);
    end;
  end
  else
    RegraUniInt[regraAleatoria] := BinToInt8(RegraBinAle);

  RegraUniBin1 := IntToBin8(RegraUniInt[0]);
  RegraUniBin2 := IntToBin8(RegraUniInt[1]);
  RegraUniBin3 := IntToBin8(RegraUniInt[2]);
  RegraUniBin4 := IntToBin8(RegraUniInt[3]);
  RegraUniBin5 := IntToBin8(RegraUniInt[4]);
  RegraUniBin6 := IntToBin8(RegraUniInt[5]);
  RegraUniBin7 := IntToBin8(RegraUniInt[6]);
  RegraUniBin8 := IntToBin8(RegraUniInt[7]);

  for I := 0 to 7 do
  begin
    strgRegra1.Cells[0,i] := IntToStr(i);
    strgRegra1.Cells[1,i] := IntToStr(RegraUniBin1[i]);
    strgRegra2.Cells[0,i] := IntToStr(i);
    strgRegra2.Cells[1,i] := IntToStr(RegraUniBin2[i]);
    strgRegra3.Cells[0,i] := IntToStr(i);
    strgRegra3.Cells[1,i] := IntToStr(RegraUniBin3[i]);
    strgRegra4.Cells[0,i] := IntToStr(i);
    strgRegra4.Cells[1,i] := IntToStr(RegraUniBin4[i]);
    strgRegra5.Cells[0,i] := IntToStr(i);
    strgRegra5.Cells[1,i] := IntToStr(RegraUniBin5[i]);
    strgRegra6.Cells[0,i] := IntToStr(i);
    strgRegra6.Cells[1,i] := IntToStr(RegraUniBin6[i]);
    strgRegra7.Cells[0,i] := IntToStr(i);
    strgRegra7.Cells[1,i] := IntToStr(RegraUniBin7[i]);
    strgRegra8.Cells[0,i] := IntToStr(i);
    strgRegra8.Cells[1,i] := IntToStr(RegraUniBin8[i]);
  end;

  n:= StrToInt(nentropia.text);

  for row:= 0 to n do
  begin
    global8 := 0;
    for col:= 0 to 7 do
    begin
      BinToInt8(Vizinhanca8(col,row));
      if col = 0 then
        strgACUni.cells[col,row+1]:= strgRegra1.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 1 then
        strgACUni.cells[col,row+1]:= strgRegra2.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 2 then
        strgACUni.cells[col,row+1]:= strgRegra3.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 3 then
        strgACUni.cells[col,row+1]:= strgRegra4.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 4 then
        strgACUni.cells[col,row+1]:= strgRegra5.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 5 then
        strgACUni.cells[col, row+1]:= strgRegra6.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 6 then
        strgACUni.cells[col, row+1]:= strgRegra7.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
      if col = 7 then
        strgACUni.cells[col, row+1]:= strgRegra8.cells[1,(StrToInt(strgProxEst.cells[row,col]))];
    end;
  end;

  vetEntropia[gEnt] := Entropia;
  strgEntropia.Cells[gEnt,0] := FloatToStr(vetEntropia[gEnt]);
  if gEnt = 0 then
  begin
//    chrtGraficoEntropia.Series[0].Clear;
//    chrtGraficoEntropiaMedia.Series[0].Clear;
    SetLength(ValoresAceitos, Length(ValoresAceitos) + 1);
    ValoresAceitos[Length(ValoresAceitos) - 1] := StrToFloat(strgEntropia.Cells[gEnt, 0]);
//    chrtGraficoEntropia.Series[0].Add(ValoresAceitos[Length(ValoresAceitos) - 1]);
    edtEntropia.Text := FloatToStr(ValoresAceitos[Length(ValoresAceitos) - 1]);

    iAtual := Length(ValoresAceitos);
    SomaAtual := ValoresAceitos[Length(ValoresAceitos) - 1];

    SetLength(vetMedias, Length(vetMedias) + 1);
    vetMedias[Length(ValoresAceitos) - 1] := SomaAtual;
//  MediaAtual := SomaAtual / iAtual;

    Application.ProcessMessages;
  end;

  if gEnt > 26 then
    SendMessage(strgEntropia.Handle, WM_HSCROLL, SB_LINERIGHT, 0);
  gEnt := gEnt + 1;
  Application.ProcessMessages;


end;

procedure TfrmPrincipal.Timer1Timer(Sender: TObject);
begin
  lbTime.Caption := TimeToStr(Time - TempoInicio);
end;

procedure TfrmPrincipal.FormShow(Sender: TObject);
begin
  lbData.Caption := DateToStr(Date);
  strgRegrasFixas.Cells[0,0] := '150';
  strgRegrasFixas.Cells[1,0] := '85';
  strgRegrasFixas.Cells[2,0] := '153';
  strgRegrasFixas.Cells[3,0] := '105';
  strgRegrasFixas.Cells[4,0] := '85';
  strgRegrasFixas.Cells[5,0] := '150';
  strgRegrasFixas.Cells[6,0] := '165';
  strgRegrasFixas.Cells[7,0] := '90';
end;

procedure TfrmPrincipal.rgRegrasClick(Sender: TObject);
begin
  if rgRegras.ItemIndex = 1 then
  begin
    strgRegrasFixas.Visible := true;
    strgACRegraUni.Visible := false;
    lbACInicial.Visible := true;
    strgACInicial.Visible := true;
  end
  else
  begin
    strgRegrasFixas.Visible := false;
    strgACRegraUni.Visible := true;
    lbACInicial.Visible := false;
    strgACInicial.Visible := false;
  end;
end;

procedure TfrmPrincipal.btnExportarClick(Sender: TObject);
var
  excel: variant;
  wlin, wcol : integer;
begin
  excel := CreateOleObject('Excel.Application');
  excel.Workbooks.add(1);
  excel.Cells.Select;
  excel.Selection.NumberFormat := '@';

  with strgMelhoresRegras do
  begin
    for wlin := 0 to RowCount-1 do
      for wcol := 0 to ColCount-1 do
        if Cells[wcol,wlin] <> '' then
        begin
          excel.cells[wlin+1,wcol+1] := strgMelhoresRegras.cells[wcol,wlin];
          excel.columns.AutoFit;
//          excel.cells[wcol,wlin].Select;
          excel.visible := true;
        end;
  end;
end;

procedure TfrmPrincipal.btnExpAleClick(Sender: TObject);
var
  excel: variant;
  wlin, wcol : integer;
begin
  excel := CreateOleObject('Excel.Application');
  excel.Workbooks.add(1);
  excel.Cells.Select;
  excel.Selection.NumberFormat := '@';

  with strgACUni do
  begin
    for wlin := 0 to RowCount-1 do
      for wcol := 0 to ColCount-1 do
        if Cells[wcol,wlin] <> '' then
        begin
          excel.cells[wlin+1,wcol+1] := strgACUni.cells[wcol,wlin];
          excel.columns.AutoFit;
//          excel.cells[wcol,wlin].Select;
          excel.visible := true;
        end;
  end;
end;

procedure TfrmPrincipal.btnExpMelhorClick(Sender: TObject);
var
  excel: variant;
  wlin, wcol : integer;
begin
  excel := CreateOleObject('Excel.Application');
  excel.Workbooks.add(1);
  excel.Cells.Select;
  excel.Selection.NumberFormat := '@';

  with strgMelhor do
  begin
    for wlin := 0 to RowCount-1 do
      for wcol := 0 to ColCount-1 do
        if Cells[wcol,wlin] <> '' then
        begin
          excel.cells[wlin+1,wcol+1] := strgMelhor.cells[wcol,wlin];
          excel.columns.AutoFit;
//          excel.cells[wcol,wlin].Select;
          excel.visible := true;
        end;
  end;
end;

end.
