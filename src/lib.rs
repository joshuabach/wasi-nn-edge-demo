use std::sync::Mutex;

// We need to use some functions from the bare wasi bindings
use wasi::{
    exports::http::incoming_handler::{Guest, IncomingRequest, ResponseOutparam},
    http::{proxy::export, types::ErrorCode},
};

// The rest are high-level definitions by the demo library
use wasi_nn_demo_lib::{
    http::RequestHandler,
    interface,
    nn::{GraphBuilder, GraphEncoding, Tensor},
};

// This is a failed attempt to carry state across invocations of
// `Compontent::handle`. Sadly, it does not work as it seems the
// component is reinitialized on every http request. As of the date of
// this report, the WASI-NN specification does not support an explicit
// way of carrying state. An option would be to store state on disk.
//
// The idea is to store the state in the handler object as a static
// variable. We need to guard this with a Mutex even though we have no
// real concurrency because safe Rust requires it.
static HANDLER: Mutex<HttpHandler> = Mutex::new(HttpHandler::new());

// To create a HTTP service in WASM, we need to create a type that
// represents our component that implements the `Guest` trait. We then
// need to "mark" this type using the `export!` macro provided by the
// wasi crate. The `handle` function of this struct will be invoked by
// the WASM runtime.
struct Component;
export!(Component);

impl Guest for Component {
    fn handle(request: IncomingRequest, response_outparam: ResponseOutparam) {
        // Working with the `IncomingRequest` and `ResponseOutparam`
        // types from the wasi-http is quite cumbersome. Luckily,
        // wasi_nn_demo_lib does all that for us and we only need to
        // call `handle_request` on our `HttpHandler` (as long as it
        // implements the `RequestHandler` trait, see below.

        let response = HANDLER
            // We aquire the lock for the handler ...
            .lock()
            .map_err(|e| ErrorCode::InternalError(Some(format!("Error locking state: {e}"))))
            // ... and then we call the handler function (provided by the wasi_nn_demo_lib)
            .and_then(|mut handler| handler.handle_request(request));

        // Finally (and even in the case of an error!) the result must
        // be finalized using this function from the wasi-http bindings:
        ResponseOutparam::set(response_outparam, response);
    }
}

struct HttpHandler {}

impl HttpHandler {
    const fn new() -> Self {
        Self {}
    }
}

// These constants are the parameters that are specific to the model
const MODEL_FORMAT: GraphEncoding = GraphEncoding::Onnx;
const MODEL_FILES: [&str; 1] = ["models/model.onnx"];
// The labels of the input and output tensors in the model
const INPUT_TENSOR_NAME: &str = "l_past_values_";
const OUTPUT_TENSOR_NAME: &str = "add_8";
// These last three constants make up the shape of the input tensors
// (16 batches of length 128: 16 x 128 x 1) and output tensors (16
// batches of length 24: 16 x 24 x 1)
const NUM_BATCHES: u32 = 16;
const HISTORY_LEN: u32 = 128;
const PREDICTION_LEN: u32 = 24;

impl RequestHandler for HttpHandler {
    // This function is called by the `handle_request` function which
    // we called in the `Guest::handle` implementation above. This way
    // we don't have to work with HTTP requests, but only the actual
    // data contained in the `interface::DataWindow` parameter.
    fn handle_data(
        &mut self,
        input: interface::DataWindow,
    ) -> Result<interface::InferenceResult, ErrorCode> {
        // We use the default execution target (cpu), but have to set
        // the model format and of course load the model files.
        let graph = GraphBuilder::default()
            .encoding(MODEL_FORMAT)
            .from_files(MODEL_FILES)?
            .build()?;
        let ctx = graph.init_execution_context()?;

        let input_tensor = tensor_from_data_window(input)?;

        // The model has only one input tensor and one output tensor.
        let output_tensors =
            &ctx.run([(INPUT_TENSOR_NAME, input_tensor)], &[OUTPUT_TENSOR_NAME])?;

        inference_result_from_tensor(&output_tensors[OUTPUT_TENSOR_NAME])
    }
}

// This function takes the raw data and converts it to a tensor that
// fits the model.
fn tensor_from_data_window(input: interface::DataWindow) -> Result<Tensor<f32>, ErrorCode> {
    // We need to make sure that the data is chronologically ordered
    let mut sorted_data_points: Vec<_> = input.data.values().collect();
    sorted_data_points.sort_by_key(|data_point| data_point.timestamp);

    // The model has no time features, it simply assumes that all the
    // data points are equidistant, so we just strip of all the
    // timestamps from the data and only work with the actual values.
    // A better way would be to either check that the timestamps are
    // equidistant or convert the received data series to an by
    // interpolating values to make it equidistant.
    let mut single_data_series: Vec<_> = sorted_data_points
        .into_iter()
        .filter_map(|data_point| match data_point.value {
            interface::Value::Number(num) => Some(num),
            // We simply ignore all string values, a better way would
            // be to return an error
            interface::Value::String(_) => None,
        })
        .collect();

    // No we force the length of the series to the batch size required
    // by the model. This strips it of at the end (discarding the most
    // recent values), a better way would probably be to strip of the
    // oldest values or just check that exactly 128 values have been
    // sent and return an error otherwise.
    single_data_series.resize(HISTORY_LEN as usize, 0f32);
    // The model wants 16 batches as inputs. Since we only have the
    // one, we just repeat that 16 times.
    let all_data_series = single_data_series.repeat(NUM_BATCHES as usize);
    let dims = vec![NUM_BATCHES, HISTORY_LEN, 1];

    Ok(Tensor::new(all_data_series, dims))
}

// This function takes the tensor inferred by the model and converts
// it into data that can be returned
fn inference_result_from_tensor(
    tensor: &Tensor<f32>,
) -> Result<interface::InferenceResult, ErrorCode> {
    let predictions: &[[f32; PREDICTION_LEN as usize]; NUM_BATCHES as usize] = tensor.try_into()?;

    // We only look at the first of the 16 batches
    let data_points = predictions[0]
        .into_iter()
        .map(|value| interface::DataPoint {
            quality: None,
            value: interface::Value::Number(value),
            // Instead of returning no timestamp, it would be possible
            // to calculate them based on the most recent timestamp in
            // the equidistant input data, since the model simply
            // continues the same time step length in its predictions.
            timestamp: None,
        })
        .collect();

    Ok(interface::InferenceResult::PredictedValues(data_points))
}
