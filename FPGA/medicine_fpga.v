
module medicine_fpga(
    input signed [7:0] feature,
    input signed [7:0] weight,
    input signed [7:0] bias,
    input [7:0] medicine_id,
    input [7:0] confidence,
    output signed [15:0] activation,
    output [7:0] result_id,
    output accepted
);

wire signed [15:0] product;
wire signed [15:0] sum;

assign product = feature * weight;
assign sum = product + bias;
assign activation = (sum > 0) ? sum : 16'd0;

assign accepted = (medicine_id != 0) && (confidence >= 70);
assign result_id = accepted ? medicine_id : 8'd0;

endmodule

module testbench;

reg signed [7:0] feature, weight, bias;
reg [7:0] medicine_id, confidence;
wire signed [15:0] activation;
wire [7:0] result_id;
wire accepted;

medicine_fpga uut(
    feature, weight, bias,
    medicine_id, confidence,
    activation, result_id, accepted
);

initial begin
    $display("MEDICINE IDENTIFICATION - FPGA SIMULATION");
    $display("-----------------------------------------");

    // Example neural calculation, plus real YOLO test confidence
    feature=10; weight=3; bias=2;
    medicine_id=1; confidence=99;
    #10;
    $display("Medicine: Meftal 250 DT");
    $display("Neural activation: %0d", activation);
    $display("YOLO confidence: %0d%%", confidence);
    $display("Accepted: %0d", accepted);
    $display("Output ID: %0d", result_id);

    // Low-confidence case
    feature=5; weight=-3; bias=2;
    medicine_id=2; confidence=40;
    #10;
    $display("-------------------------");
    $display("Low-confidence YOLO prediction");
    $display("Neural activation: %0d", activation);
    $display("YOLO confidence: %0d%%", confidence);
    $display("Accepted: %0d", accepted);

    // Untrained medicine: no verified class ID
    feature=8; weight=2; bias=1;
    medicine_id=0; confidence=0;
    #10;
    $display("-------------------------");
    $display("Medicine: Neurobion (untrained)");
    $display("Neural activation: %0d", activation);
    $display("Accepted: %0d", accepted);
    $display("OCR identification handled by software");

    $finish;
end
endmodule
