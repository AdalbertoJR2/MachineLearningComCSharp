using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class CasaModelTrainer
{
    private MLContext mlContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;
    public void CarregarDadosCSV(string path)
    {
        dados = mlContext.Data.LoadFromTextFile<CasaInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        var pipeline = mlContext.Transforms.Concatenate(
            "Features",
            nameof(CasaInputData.Tamanho),
            nameof(CasaInputData.Quartos)
        )/*.Append(mlContext.Regression.Trainers.Sdca(
            labelColumnName: "Preco",
            maximumNumberOfIterations: 100
        ));*/
        .Append(mlContext.Regression.Trainers.LightGbm(
            labelColumnName: "Preco",
            numberOfIterations: 100
        ));

        modeloTreinado = pipeline.Fit(dados);
    }

    public void SalvarModelo(string path)
    {
        mlContext.Model.Save(modeloTreinado, dados.Schema, path);
    }

    public void AvaliarModelo()
    {
        var previcoes = modeloTreinado.Transform(dados);

        var metricas = mlContext.Regression.Evaluate(
            data: previcoes,
            labelColumnName: "Preco",
            scoreColumnName: "Score"
        );

        Console.WriteLine($"MAE: {metricas.MeanAbsoluteError}");
        Console.WriteLine($"RMSE: {metricas.RootMeanSquaredError}");
        Console.WriteLine($"R2: {metricas.RSquared}");
    }
}
