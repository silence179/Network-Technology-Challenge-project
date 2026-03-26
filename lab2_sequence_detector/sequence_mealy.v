// sequence_mealy.v
// Mealy FSM: 时序检测器（连续3个1检测）
// 输出取决于当前状态和当前输入：
//   在S2或S3状态下，若输入为1则输出1；否则输出0
// 状态编码：
//   S0 = 2'b00  已检测到0个连续1
//   S1 = 2'b01  已检测到1个连续1
//   S2 = 2'b10  已检测到2个连续1
//   S3 = 2'b11  已检测到3+个连续1

module sequence_mealy (
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

    // 输出逻辑（Mealy：取决于当前状态和当前输入）
    always @(*) begin
        case (state)
            S2:      dout = din ? 1'b1 : 1'b0;
            S3:      dout = din ? 1'b1 : 1'b0;
            default: dout = 1'b0;
        endcase
    end

endmodule
