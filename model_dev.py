"""Module to build and compare models."""
import pandas as pd
import numpy as np 
import geopandas as gpd
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.base import clone
import joblib  # For saving the model




warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def prepare_data(gdf, cols_to_remove, crs = "EPSG:4326"):
    """
    This function removes unnecessary columns and normalizes remaining features by 
    area of census subdivision. 

    Parameters:
        df: gdf, assume contains features and target
        cols_to_remove: list
    """

    # A. Project to new CRS
    gdf_final = gdf.to_crs(crs)

    # # B. Remove Yukon
    # gdf_no_yukon = gdf_proj[gdf_proj["Subdivision"]!="Yukon"].copy()
    subdivisions = gdf_final["csd_name_en"]
    provinces  = gdf_final["Province"]

    # C. Convert area unit from m to km
    gdf_final["area_km_sq"]=gdf_final.area/10**6
    areas = gdf_final["area_km_sq"]

    # D. Drop columns
    gdf_final = gdf_final.drop(columns=cols_to_remove) 

    # E. Find normalization. NOTE: This makes our response pop density, not pop count. 
    gdf_final = gdf_final.apply(lambda row: row / row["area_km_sq"], axis=1)
    gdf_final["population_density"] = gdf_final["population"]

    # F. Remove area
    gdf_final = gdf_final.drop(columns=["area_km_sq", "population"])
    
    gdf_final["Province"] = provinces

    gdf_final = gdf_final.dropna()

    return(gdf_final, areas, subdivisions)

import warnings
from sklearn.exceptions import ConvergenceWarning

import warnings
from sklearn.exceptions import ConvergenceWarning

def compare_models_logo_warnings(regressors, gdf, degrees=[1, 2], cv_strategy=None,  save_path="best_model.pkl"):
    """
    Compares different models using MAE and optionally includes polynomial features.

    Parameters:
        regressors: dict of regressors
        gdf: GeoDataFrame with features and target
        degrees: list of polynomial degrees to try (default=[1, 2])
        cv_strategy: cross-validation object (e.g., LeaveOneGroupOut or custom spatial CV)

    Returns:
        best_model: The best-performing model (fitted on the training data)
        best_params: Parameters of the best model
        results: DataFrame with performance metrics and warnings for all models
    """

    # Prepare target and features
    y = gdf["population_density"]
    X = gdf.drop(columns=["population_density", "Province"])  # Drop Province directly

    # Extract groups if the cross-validation strategy requires it
    groups = gdf["Province"] if hasattr(cv_strategy, 'split') and 'groups' in cv_strategy.split.__code__.co_varnames else None

    # Initialize variables to track the best model
    best_model = None
    best_params = None
    best_mae = float("inf")
    results = []

    for degree in degrees:
        # Precompute polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        for name, model in regressors.items():
            warning_raised = None  # Initialize warning tracker
            try:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", clone(model))
                ])

                # Suppress warnings during cross-validation
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # Capture all warnings
                    scores = cross_val_score(
                        pipeline, X_poly, y, cv=cv_strategy,
                        scoring='neg_mean_absolute_error', groups=groups, n_jobs=1
                    )

                    # Check if any warnings were raised
                    if len(w) > 0:
                        warning_raised = "; ".join([str(warning.message) for warning in w])

                # Calculate mean MAE
                mae = -np.mean(scores)

                # Optionally fit on all data for consistent prediction
                pipeline.fit(X_poly, y)
                current_params = {"model": name, "degree": degree}

                # Save results
                results.append({
                    "Model": name,
                    "Degree": degree,
                    "MAE": mae,
                    "Warning": warning_raised  # Store warnings in the results
                })

                # Debug output
                print(f"Evaluated {name} with degree {degree} => MAE: {mae} (current best: {best_mae})")

                # Update the best model if this one is better
                if np.isfinite(mae) and mae < best_mae:
                    print(f"  → Updating best model to {name} (degree {degree}) with MAE {mae}")
                    best_mae = mae
                    best_model = pipeline
                    best_params = current_params

                    # Save the best model to a file
                    joblib.dump(best_model, save_path)
                    print(f"Best model saved to {save_path}")


            except Exception as e:
                warning_raised = str(e)  # Store exception as a warning
                print(f"Model {name} with degree {degree} failed: {e}")
                results.append({
                    "Model": name,
                    "Degree": degree,
                    "MAE": np.nan,
                    "Warning": warning_raised
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    print("Best Model:", best_params)
    print("Lowest MAE in results_df:", results_df["MAE"].min())
    return best_model, best_params, results_df

def compare_models_logo(regressors, gdf, degrees=[1, 2], cv_strategy=None):
    

    y = gdf["population_density"]
    X = gdf.drop(columns=["population_density", "Province"])

    groups = gdf["Province"] if hasattr(cv_strategy, 'split') and 'groups' in cv_strategy.split.__code__.co_varnames else None

    best_model = None
    best_params = None
    best_mae = float("inf")
    results = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        for name, model in regressors.items():
            try:
                if name in ['RidgeCV', 'LassoCV', 'ElasticNetCV']:
                    param_grid = {}
                    if name == 'RidgeCV':
                        base_model = Ridge()
                        param_grid = {'alpha': model.alphas}
                    elif name == 'LassoCV':
                        base_model = Lasso(max_iter=model.max_iter)
                        param_grid = {'alpha': model.alphas}
                    elif name == 'ElasticNetCV':
                        base_model = ElasticNet(max_iter=model.max_iter)
                        param_grid = {'alpha': model.alphas, 'l1_ratio': model.l1_ratio}

                    grid = GridSearchCV(
                        base_model, param_grid, cv=cv_strategy,
                        scoring='neg_mean_absolute_error', n_jobs=-1
                    )
                    grid.fit(X_poly, y, groups=groups)

                    mae = -grid.best_score_
                    best_pipeline = grid.best_estimator_
                    current_params = {"model": name, "degree": degree, **grid.best_params_}

                else:
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("model", clone(model))
                    ])
                    scores = cross_val_score(
                        pipeline, X_poly, y, cv=cv_strategy,
                        scoring='neg_mean_absolute_error', groups=groups, n_jobs=-1
                    )
                    mae = -np.mean(scores)
                    pipeline.fit(X_poly, y)
                    best_pipeline = pipeline
                    current_params = {"model": name, "degree": degree}

                results.append({
                    "Model": name,
                    "Degree": degree,
                    "MAE": mae
                })

                # Debug output
                print(f"Evaluated {name} with degree {degree} => MAE: {mae} (current best: {best_mae})")

                if np.isfinite(mae) and mae < best_mae:
                    print(f"  → Updating best model to {name} (degree {degree}) with MAE {mae}")
                    best_mae = mae
                    best_model = best_pipeline
                    best_params = current_params

            except Exception as e:
                print(f"Model {name} with degree {degree} failed: {e}")

    results_df = pd.DataFrame(results)
    print("Best Model:", best_params)
    print("Lowest MAE in results_df:", results_df["MAE"].min())
    return best_model, best_params, results_df

def compare_models(regressors, gdf, degrees=[1, 2]):
    """
    Compares different models using MAE and optionally includes polynomial features.

    Parameters:
        regressors: dict of regressors
        gdf: GeoDataFrame with features and target
        degrees: list of polynomial degrees to try (default=[1, 2])
    """
    data = []
   
    y = gdf["population_density"]
    X = gdf.drop(columns=["population_density"])
    loo = LeaveOneOut()

    for name, model in regressors.items():
        for degree in degrees:
            # Create a pipeline with optional polynomial features
            pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),  # Add polynomial terms
                ("scaler", StandardScaler()),  # Ensure scaling after polynomial transformation
                ("regressor", model)
            ])

            # Track convergence and warnings
            converged = True
            warning_raised = None
            best_params = None
            try:
                if name in ['LinearRegression', 'RandomForest', 'GradientBoosting']:
                    # Use cross_val_predict for models that don't support built-in CV
                    y_pred = cross_val_predict(pipeline, X, y, cv=loo)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=ConvergenceWarning)  # Treat ConvergenceWarning as an error
                        warnings.filterwarnings("error", category=UndefinedMetricWarning)  # Treat UndefinedMetricWarning as an error
                        warnings.filterwarnings("error", category=UserWarning)  # Treat UserWarning as an error
                        pipeline.fit(X, y)
                        y_pred = pipeline.predict(X)

                    # Extract best hyperparameters for models that support it
                    if hasattr(model, "alpha_"):  # For RidgeCV, LassoCV, ElasticNetCV
                        best_params = f"alpha={model.alpha_}"
                    if hasattr(model, "l1_ratio_"):  # For ElasticNetCV
                        best_params += f", l1_ratio={model.l1_ratio_}"

            except ConvergenceWarning:
                converged = False
                warning_raised = "ConvergenceWarning"
                y_pred = [0] * len(y)  # Set predictions to 0 if the model fails to converge
            except UndefinedMetricWarning:
                converged = False
                warning_raised = "UndefinedMetricWarning"
                y_pred = [0] * len(y)  # Handle undefined metrics gracefully
            except UserWarning:
                converged = False
                warning_raised = "UserWarning"
                y_pred = [0] * len(y)  # Handle user warnings gracefully

            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            new_row = [name, degree, mae, mse, converged, warning_raised, best_params]
            data.append(new_row)
    
    # Create DataFrame with results
    df_to_return = pd.DataFrame(data, columns=["Regressor", "Polynomial Degree", "MAE", "MSE", "Converged", "Warning", "Best Params"]).sort_values(by="MAE", ascending=True)
    return df_to_return



if __name__ == "__main__":
    gdf = gpd.read_file("Data/osm_extractor_output")

    cols_to_remove = ["geo_point_2d",
                    "year",
                    "prov_code",
                    "prov_name_en",
                    "cd_code",
                    "cd_name_en",
                    "csd_code",
                    "csd_name_en",
                    "csd_area_code",
                    "csd_type",
                    "prov_name_fr",
                    "cd_name_fr",
                    "csd_name_fr",
                    "Subdivision",
                    "updated",
                    "geometry"]
    gdf_norm, areas, subdivisions = prepare_data(gdf, cols_to_remove, crs='epsg:3857')

        # Leave-One-Out CV
    loo = LeaveOneOut()

    regressors = {
        'LinearRegression': LinearRegression(),
        'RidgeCV': RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=loo),
        'LassoCV': LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=loo, max_iter=10000),
        'ElasticNetCV': ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10],
                                     l1_ratio=[0.1, 0.5, 0.9],
                                     cv=loo,
                                     max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                      max_depth=3, random_state=42)
    }

    performance = compare_models(regressors, gdf_norm, degrees=[1, 2])
    print(performance)


        

    

    














