package com.example.myapplication.models
import java.io.Serializable
data class user(
    val id: String? = null,
    val password: String? = null,
    val age: String = "",
    val ranRangeNum: String? = "3",
    val level: String? = "1",
    val streak: String? = "0",
    val session_id: String? = "1",
) : Serializable
