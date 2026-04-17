using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class CasaModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    private ITransformer modeloTreinado;
    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<CasaInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }

    public void TreinarModelo()
    {
        var pipeline = mLContext.Transforms.Concatenate(
            "Features",
            nameof(CasaInputData.Tamanho),
            nameof(CasaInputData.Quartos)
        ).Append(mLContext.Regression.Trainers.Sdca(
            labelColumnName: "Preco",
            maximumNumberOfIterations: 100
        ));

        modeloTreinado = pipeline.Fit(dados);
    }

    public void SalvarModelo(string path)
    {
        mLContext.Model.Save(modeloTreinado, dados.Schema, path);
    }
}
