import axios from "axios";

export const customAxios = axios.create({
  baseURL: "192.168.0.68:6002"
});