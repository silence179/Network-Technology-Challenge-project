// sequence_moore.v
// Moore FSM: 时序检测器（连续3个1检测）
// 状态输出：S3时输出1，其余状态输出0
// 状态编码：
//   S0 = 2'b00  已检测到0个连续1
//   S1 = 2'b01  已检测到1个连续1
//   S2 = 2'b10  已检测到2个连续1
//   S3 = 2'b11  已检测到3个连续1（输出1）

module sequence_moore (
    input  wire clk,
    input  wire rst,
    input  wire din,
    output reg  dout
);

    parameter S0 = 2'b00;
    parameter S1 = 2'b01;
    parameter S2 = 2'b10;
    parameter S3 = 2'b11;

    reg [1:0] state;

    // 状态寄存器（异步复位）
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= S0;
        else
            case (state)
                S0: state <= din ? S1 : S0;
                S1: state <= din ? S2 : S0;
                S2: state <= din ? S3 : S0;
                S3: state <= din ? S3 : S0;
                default: state <= S0;
            endcase
    end

    // 输出逻辑（Moore：仅取决于当前状态）
    always @(*) begin
        case (state)
            S3:      dout = 1'b1;
            default: dout = 1'b0;
        endcase
    end

endmodule
